from functools import partial
from jax import random
import jax
import jax.numpy as np
import orbax.checkpoint as ocp
import os
from jax.scipy.linalg import block_diag
import wandb

from .train_helpers import (
    create_train_state,
    reduce_lr_on_plateau,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    train_epoch,
    validate,
)
from .dataloading import Datasets
from .qseq_model import QBatchClassificationModel, QRetrievalModel
from .qssm_aqt import init_qS5SSM, QuantizationConfig
from .ssm_init import make_DPLR_HiPPO


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in [
        "imdb-classification",
        "listops-classification",
        "aan-classification",
    ]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # For speech dataset
    if args.dataset in ["speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    q_config = QuantizationConfig(
        a_precision=args.a_bits,
        b_precision=args.b_bits,
        c_precision=args.c_bits,
        d_precision=args.d_bits,
        non_ssm_precision=args.non_ssm_bits,
        ssm_act_precision=args.ssm_act_bits,
        non_ssm_act_precision=args.non_ssm_act_bits,
    )
    ssm_init_fn = init_qS5SSM(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
        q_config=q_config,
    )

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            QRetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            q_bits_aw=(q_config.non_ssm_act_precision, q_config.non_ssm_precision),
            use_hard_sigmoid=args.hard_sigmoid,
            use_q_gelu_approx=args.qgelu_approx,
            use_layernorm_bias=args.use_layernorm_bias,
        )

    else:
        model_cls = partial(
            QBatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            q_bits_aw=(q_config.non_ssm_act_precision, q_config.non_ssm_precision),
            use_hard_sigmoid=args.hard_sigmoid,
            use_q_gelu_approx=args.qgelu_approx,
            use_layernorm_bias=args.use_layernorm_bias,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        grad_clip_threshold=args.grad_clip_threshold,
        dt_global=args.dt_global,
    )

    # Setup checkpointing & wandb
    chkpt_metadata = {
        "best_test_loss": 100000000,
        "best_test_acc": -10000.0,
        "wandb_id": None,
        "last_step": 0,
        "next_epoch": 0,
    }
    restored_state = None

    # create checkpoint manager
    chkpt_mngr = None
    if args.load_run_name is not None and args.checkpoint_dir is not None:
        # create directory for model checkpoints
        chkpt_path = os.path.join(args.checkpoint_dir, args.load_run_name)
        os.makedirs(chkpt_path, exist_ok=True)

        # create checkpoint manager
        chkpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_interval_steps,
            max_to_keep=args.checkpoint_max_to_keep,
        )
        chkpt_mngr = ocp.CheckpointManager(
            directory=chkpt_path,
            item_names=("state", "metadata"),
            options=chkpt_options,
        )
        
        # check if we should load a checkpoint
        if chkpt_mngr.latest_step() is not None:
            abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
            restored = chkpt_mngr.restore(
                chkpt_mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_state),
                    metadata=ocp.args.JsonRestore(),
                )
            )
            restored_state = restored["state"]
        
        # remove this checkpoint manager (avoid overwriting)
        chkpt_mngr = None

    if args.run_name is not None and args.checkpoint_dir is not None:
        # create directory for model checkpoints
        chkpt_path = os.path.join(args.checkpoint_dir, args.run_name)
        os.makedirs(chkpt_path, exist_ok=True)

        # create checkpoint manager
        chkpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_interval_steps,
            max_to_keep=args.checkpoint_max_to_keep,
        )
        chkpt_mngr = ocp.CheckpointManager(
            directory=chkpt_path,
            item_names=("state", "metadata"),
            options=chkpt_options,
        )

        # check if we should load a checkpoint (make sure we didn't already load one above)
        if chkpt_mngr.latest_step() is not None and restored_state is None:
            abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
            restored = chkpt_mngr.restore(
                chkpt_mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_state),
                    metadata=ocp.args.JsonRestore(),
                )
            )
            chkpt_metadata = restored["metadata"]
            restored_state = restored["state"]

    # Setup wandb
    if args.wandb_apikey is not None:
        wandb.login(key=args.wandb_apikey)

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
            name=args.run_name,
            id=chkpt_metadata["wandb_id"],
            resume="allow",  # if run_id doesn't exist, make a new run
        )
        chkpt_metadata["wandb_id"] = wandb.run.id
    else:
        wandb.init(mode="offline")

    wandb.log({"block_size": block_size})

    # Restore training state and other metadata
    if restored_state is not None:
        print("\nRestoring train state from checkpoint...\n")
        state = restored_state

    best_test_loss = chkpt_metadata["best_test_loss"]
    best_test_acc = chkpt_metadata["best_test_acc"]
    step = chkpt_metadata["last_step"]
    epoch_start = chkpt_metadata["next_epoch"]

    # Training loop over epochs
    best_loss, best_acc, best_epoch = (
        100000000,
        -100000000.0,
        0,
    )  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    steps_per_epoch = int(train_size / args.bsz)
    for epoch in range(epoch_start, args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (
                steps_per_epoch * args.warmup_end
            )
        else:
            print("using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (
            decay_function,
            ssm_lr,
            lr,
            step,
            end_step,
            args.opt_config,
            args.lr_min,
        )

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state,
            skey,
            model_cls,
            trainloader,
            seq_len,
            in_dim,
            args.batchnorm,
            lr_params,
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            # adding the prng key so that aqt/flax doesn't complain
            val_loss, val_acc = validate(
                state, skey, model_cls, valloader, seq_len, in_dim, args.batchnorm
            )

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, skey, model_cls, testloader, seq_len, in_dim, args.batchnorm
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
                f" --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(
                state, skey, model_cls, testloader, seq_len, in_dim, args.batchnorm
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # Do some validation on improvement.
            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc = validate(
                    state,
                    model_cls,
                    aux_dataloaders["valloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                )

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc = validate(
                    state,
                    model_cls,
                    aux_dataloaders["testloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                )
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
            input,
            factor=args.reduce_factor,
            patience=args.lr_patience,
            lr_min=args.lr_min,
        )

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if valloader is not None:
            if speech:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Val2 loss": val2_loss,
                        "Val2 Accuracy": val2_acc,
                        "Test2 Loss": test2_loss,
                        "Test2 Accuracy": test2_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states[
                            "regular"
                        ].inner_state.hyperparams["learning_rate"],
                        "ssm_lr": state.opt_state.inner_states[
                            "ssm"
                        ].inner_state.hyperparams["learning_rate"],
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states[
                            "regular"
                        ].inner_state.hyperparams["learning_rate"],
                        "ssm_lr": state.opt_state.inner_states[
                            "ssm"
                        ].inner_state.hyperparams["learning_rate"],
                    }
                )

        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": state.opt_state.inner_states[
                        "regular"
                    ].inner_state.hyperparams["learning_rate"],
                    "ssm_lr": state.opt_state.inner_states[
                        "ssm"
                    ].inner_state.hyperparams["learning_rate"],
                }
            )
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        # save checkpoint
        if chkpt_mngr is not None:
            chkpt_metadata["best_test_loss"] = best_test_loss.item()
            chkpt_metadata["best_test_acc"] = best_test_acc.item()
            chkpt_metadata["last_step"] = step
            chkpt_metadata["next_epoch"] = epoch + 1
            chkpt_mngr.save(
                step=epoch+1,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    metadata=ocp.args.JsonSave(chkpt_metadata),
                )
            )
            chkpt_mngr.wait_until_finished()

        if count > args.early_stop_patience:
            break
