from functools import partial
from flax.training import train_state
from jax import random
import jax
import jax.numpy as np
import orbax.checkpoint as ocp
import os
from jax.scipy.linalg import block_diag
import wandb

from .train_helpers import (
    create_train_state,
    validate,
)
from .dataloading import Datasets
from .qseq_model import QBatchClassificationModel, QRetrievalModel
from .qssm_aqt import init_qS5SSM, QuantizationConfig
from .ssm_init import make_DPLR_HiPPO


def evaluate(args):
    """
    Main function to evaluate a model on a dataset.
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
            use_qlayernorm_if_quantized=args.use_qlayernorm_if_quantized,
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
            use_qlayernorm_if_quantized=args.use_qlayernorm_if_quantized,
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
        print("loading checkpoint:", args.load_run_name, args.checkpoint_dir)
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
            if args.remove_norm_bias_from_checkpoint:
                # NOTE: this hack was only tested for sMNIST
                print("attempting to remove norm bias from checkpoint")
                newstateparams = jax.tree_util.tree_map(
                    lambda x: x if not hasattr(x, 'keys') else {'scale': x['scale'], 'bias': x.get('bias', x['scale'])},
                    state.params,
                    is_leaf=lambda x: not hasattr(x, 'keys') or 'scale' in x.keys()
                )
                state = train_state.TrainState.create(apply_fn=state.apply_fn, params=newstateparams, tx=state.tx)

            abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
            restored = chkpt_mngr.restore(
                chkpt_mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_state),
                    metadata=ocp.args.JsonRestore(),
                )
            )

            chkpt_metadata = restored["metadata"]
            print(chkpt_metadata)

            print("\nRestoring train state from checkpoint...")
            state = restored["state"]
            if args.remove_norm_bias_from_checkpoint:
                # NOTE: this hack was only tested for sMNIST
                print("removing norm bias from checkpoint")
                newstateparams = jax.tree_util.tree_map(
                    lambda x: x if not hasattr(x, 'keys') else {'scale': x['scale']},
                    state.params,
                    is_leaf=lambda x: not hasattr(x, 'keys') or 'scale' in x.keys()
                )
                state = train_state.TrainState.create(apply_fn=state.apply_fn, params=newstateparams, tx=state.tx)
        else:
            print(f"\n[WARNING] no checkpoint found for {args.load_run_name}!!\n")
    else:
        print("\n[WARNING] running evaluation without loading a checkpoint!!\n")

    # # Setup wandb
    # if args.wandb_apikey is not None:
    #     wandb.login(key=args.wandb_apikey)
    # if args.USE_WANDB:
    #     # Make wandb config dictionary
    #     wandb.init(
    #         project=args.wandb_project,
    #         job_type="model_training",
    #         config=vars(args),
    #         entity=args.wandb_entity,
    #         name=args.run_name,
    #     )
    # else:
    #     wandb.init(mode="offline")
    # wandb.log({"block_size": block_size})
    # best_test_loss = chkpt_metadata["best_test_loss"]
    # best_test_acc = chkpt_metadata["best_test_acc"]

    train_rng, skey = random.split(train_rng)
    train_loss, train_acc = validate(
        state, skey, model_cls, trainloader, seq_len, in_dim, args.batchnorm
    )

    test_loss, test_acc = None, None
    if valloader is not None:
        print(f"[*] Running Validation...")
        # adding the prng key so that aqt/flax doesn't complain
        val_loss, val_acc = validate(
            state, skey, model_cls, valloader, seq_len, in_dim, args.batchnorm
        )

        print(f"[*] Running Test...")
        test_loss, test_acc = validate(
            state, skey, model_cls, testloader, seq_len, in_dim, args.batchnorm
        )

        print(args.run_name)

        print(f"\n=>> Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
            f" --Test Loss: {test_loss:.5f} --"
            f" Train Accuracy: {train_acc:.4f} Val Accuracy: {val_acc:.4f}"
            f" Test Accuracy: {test_acc:.4f}"
        )

    else:
        # else use test set as validation set (e.g. IMDB)
        print(f"[*] Running Test...")
        val_loss, val_acc = validate(
            state, skey, model_cls, testloader, seq_len, in_dim, args.batchnorm
        )

        print(f"\n=>> Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
            f" Train Accuracy: {train_acc:.4f} Test Accuracy: {val_acc:.4f}"
        )

    folder = "/home/sabreu/NeuroSSMs/results/PTQ/"
    os.makedirs(folder, exist_ok=True)
    fname = f"ptq_results_{args.run_name}.txt"
    with open(os.path.join(folder, fname), "w") as f:
        header = "train_loss,val_loss,test_loss,train_acc,val_acc,test_acc"
        f.write(f"{header}\n{train_loss},{val_loss},{test_loss},{train_acc},{val_acc},{test_acc}\n")

    # if valloader is not None:
    #     wandb.log(
    #         {
    #             "Training Loss": train_loss,
    #             "Training Accuracy": train_acc,
    #             "Val loss": val_loss,
    #             "Val Accuracy": val_acc,
    #             "Test Loss": test_loss,
    #             "Test Accuracy": test_acc,
    #         }
    #     )
    # else:
    #     wandb.log(
    #         {
    #             "Training Loss": train_loss,
    #             "Training Accuracy": train_acc,
    #             "Val loss": val_loss,
    #             "Val Accuracy": val_acc,
    #         }
    #     )
    # wandb.run.summary["Training Loss"] = train_loss
    # wandb.run.summary["Training Accuracy"] = train_acc
    # wandb.run.summary["Val Loss"] = val_loss
    # wandb.run.summary["Val Accuracy"] = val_acc
    # wandb.run.summary["Test Loss"] = test_loss
    # wandb.run.summary["Test Accuracy"] = test_acc
    # wandb.run.summary["Best Test Loss"] = best_test_loss
    # wandb.run.summary["Best Test Accuracy"] = best_test_acc
