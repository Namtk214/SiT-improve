import os
import argparse
import pickle
import time
import threading
import queue

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state, checkpoints
from flax import jax_utils
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import torch

# Import data loaders
import tensorflow as tf
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    print("WARNING: grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")

from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img, scattercat


def create_train_state(rng, config, learning_rate):
    """Initializes the model and TrainState."""
    model = SelfFlowPerTokenDiT(
        input_size=config["input_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=config["num_classes"],
        learn_sigma=config["learn_sigma"],
        compatibility_mode=config["compatibility_mode"],
        per_token=True,
    )

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)
    
    rng, drop_rng = jax.random.split(rng)
    variables = model.init(
        {'params': rng, 'dropout': drop_rng}, 
        x=dummy_x, 
        timesteps=dummy_t, 
        vector=dummy_vec, 
        deterministic=False
    )
    
    tx = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )


@jax.pmap
def train_step(state, batch, rng):
    """Executes a single distributed training step."""
    x, y = batch
    
    rng, step_rng, time_rng, noise_rng, drop_rng = jax.random.split(rng, 5)
    
    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    noise = jax.random.normal(noise_rng, x.shape)
    
    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x 
    target = x - noise
    
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            x_t,
            timesteps=t,
            vector=y,
            deterministic=False,
            rngs={'dropout': drop_rng}
        )
        # Compute losses
        loss_sq = (pred - target) ** 2
        loss = jnp.mean(loss_sq)
        
        # Internal Metrics calculation to avoid host transfers
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        
        return loss, (v_abs_mean, v_pred_abs_mean)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_abs, v_pred)), grads = grad_fn(state.params)
    
    # Cross-device synchronization (TPU v5e-8 Data Parallel)
    loss = jax.lax.pmean(loss, axis_name='batch')
    v_abs = jax.lax.pmean(v_abs, axis_name='batch')
    v_pred = jax.lax.pmean(v_pred, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Calculate norms on device
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)]))
    param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)]))
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        "train/loss": loss,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    
    return state, metrics, rng


def get_arrayrecord_dataloader(data_pattern, batch_size, is_training=True, seed=42):
    """
    Creates an optimized Grain dataloader reading from ArrayRecord files.
    """
    data_source = grain.ArrayRecordDataSource(data_pattern)
    
    class ParseAndTokenizeLatents(grain.MapTransform):
        def map(self, record_bytes):
            parsed = pickle.loads(record_bytes)
            
            latent = parsed["latent"] # numpy array shape: (4, 32, 32)
            label = parsed["label"]
            
            # Patchify the latent to DiT input (256, 16)
            c, h, w = latent.shape
            p = 2
            
            # Using numpy to manipulate shapes to send cleanly into DataLoader
            latent = np.reshape(latent, (c, h // p, p, w // p, p))
            latent = np.transpose(latent, (1, 3, 2, 4, 0)) # block arrangement
            latent = np.reshape(latent, ((h // p) * (w // p), p * p * c))
            
            return latent, label
            
    operations = [
        ParseAndTokenizeLatents(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None if is_training else 1,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=is_training,
        seed=seed,
    )

    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=8,
        read_options=grain.ReadOptions(prefetch_buffer_size=1024)
    )
    
    return dataloader


class AsyncWandbLogger:
    """Background thread to log metrics without blocking TPU pipeline."""
    def __init__(self, max_queue_size=50):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
                
            metrics, step = item
            
            # Perform jax.device_get to block *only* the worker thread
            try:
                metrics_cpu = jax.tree_util.tree_map(lambda x: float(x) if hasattr(x, 'shape') and x.shape == () else x, jax.device_get(metrics))
                wandb.log(metrics_cpu, step=step)
            except Exception as e:
                print(f"WandB Logging error: {e}")
            finally:
                self.queue.task_done()
                
    def log(self, metrics, step):
        try:
            # We use put_nowait so if the queue backs up, we just drop logs rather than stalling TPU
            self.queue.put_nowait((metrics, step))
        except queue.Full:
            pass # Skip logging if CPU is lagging too far behind TPU
            
    def shutdown(self):
        self.queue.put(None)
        self.thread.join()

  
@jax.jit
def sample_latents_jit(params, class_labels, rng, num_steps=50, cfg_scale=4.0):
    """Generate sample latents on TPU."""
    batch_size = class_labels.shape[0]
    latent_channels, latent_size, patch_size = 4, 32, 2
    
    noise = jax.random.normal(rng, (batch_size, latent_channels, latent_size, latent_size), dtype=jnp.bfloat16)
    
    from einops import rearrange
    noise_patched = rearrange(noise, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=patch_size, p2=patch_size)
    x, x_ids = batched_prc_img(noise_patched)
    
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        x = jnp.concatenate([x, x], axis=0)
        x_ids = jnp.concatenate([x_ids, x_ids], axis=0)
        class_labels = jnp.concatenate([jnp.full_like(class_labels, 1000), class_labels], axis=0)
        
    def model_fn(z_x, t):
        # We need a dummy apply_fn call mapping mechanism
        model = SelfFlowPerTokenDiT(
            input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, 
            num_heads=16, mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True, per_token=True
        )
        return model.apply({'params': params}, z_x, timesteps=t, vector=class_labels, deterministic=True)
        
    rng, denoise_rng = jax.random.split(rng)
    samples = denoise_loop(
        model_fn=model_fn, x=x, rng=denoise_rng, num_steps=num_steps,
        cfg_scale=cfg_scale, guidance_low=0.0, guidance_high=0.7, mode="SDE"
    )
    
    if use_cfg:
        samples = samples[batch_size:]
        x_ids = x_ids[batch_size:]
        
    samples = scattercat(samples, x_ids)
    samples = rearrange(samples, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, c=latent_channels)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Self-Flow DiT (JAX)")
    parser.add_argument("--batch-size", type=int, default=256, help="Global Batch size (will be divided by 8 for TPU v5e-8)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of steps in an epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data-path", type=str, default="/path/to/imagenet/latents/*.ar", help="Path to ArrayRecords")
    parser.add_argument("--wandb-project", type=str, default="selfflow-jax", help="WandB Project Name")
    parser.add_argument("--log-freq", type=int, default=20, help="Log step metrics every N steps")
    parser.add_argument("--sample-freq", type=int, default=1000, help="Generate and decode samples every M steps")
    args = parser.parse_args()
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    logger = AsyncWandbLogger()

    # Device count checks
    num_devices = jax.device_count()
    local_batch_size = args.batch_size // num_devices
    print(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    rng = jax.random.PRNGKey(42)
    
    config = dict(
        input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28,
        num_heads=16, mlp_ratio=4.0, num_classes=1001, learn_sigma=True, compatibility_mode=True,
    )
    
    state = create_train_state(rng, config, args.learning_rate)
    # Replicate state across all TPU cores
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, num_devices)
    
    print("Initialized Replicated TrainState")
    
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    try:
        dataloader = get_arrayrecord_dataloader(data_pattern=args.data_path, batch_size=args.batch_size, is_training=True)
        data_iterator = iter(dataloader)
        print("DataLoader initialized successfully via Grain.")
    except Exception as e:
        print(f"Failed to load ArrayRecord via Grain. Falling back to mocked batches. Error: {e}")
        data_iterator = None

    # Load SD-VAE exclusively on CPU (Host) for standalone asynchronous decoding. 
    # This prevents blocking the main TPU SPMD devices.
    print("Loading VAE on Host CPU for WandB image generation...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae.eval()
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            if data_iterator is not None:
                # Real TPU Batch from ArrayRecord Pipeline
                batch = next(data_iterator)
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback
                rng_mock, = jax.random.split(rng[0], 1)
                batch_x = jax.random.normal(rng_mock, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(rng_mock, (args.batch_size,), 0, 1000)
            
            # Reshape batch for SPMD distribution: (Global, ...) -> (Devices, Local, ...)
            batch_x = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
            batch_y = batch_y.reshape(num_devices, local_batch_size)
            
            # Pmap execute step
            state, metrics, rng = train_step(state, (batch_x, batch_y), rng)
            global_step += 1
            
            # Periodic Async Logging
            if global_step % args.log_freq == 0:
                # Extract index 0 since pmap returns duplicated metrics for all cores
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["train/step"] = global_step
                t0 = time.time()
                
                logger.log(cpu_metrics, step=global_step)
            
            # Periodic Image Evaluation & Generation (Latents generated on TPU[0], Decoded on CPU Thread via VAE)
            if global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating evaluation samples...")
                
                # Use ONLY Core 0 explicitly to generate Latents
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                
                # Fetch params of Core 0
                single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
                
                # Generate latents asynchronously via JIT
                latents_dev = sample_latents_jit(single_params, sample_classes, sample_rng)
                
                # Hand over to background worker to pull array to host, decode in PyTorch VAE, and wandb.log
                def background_decode_and_log(z_dev, classes, target_step):
                    # Blocking device_get ONLY on this temporary background thread
                    z = jax.device_get(z_dev) 
                    z = torch.from_numpy(z)
                    classes = jax.device_get(classes)
                    
                    z = z / 0.18215 # Scale factor
                    with torch.no_grad():
                        images = vae.decode(z).sample
                        
                    images = (images + 1.0) / 2.0
                    images = images.clamp(0, 1).permute(0, 2, 3, 1).numpy()
                    images = (images * 255).astype(np.uint8)
                    
                    wandb.log({
                        "train/step": target_step,
                        "samples": [wandb.Image(img, caption=f"Class {cls}") for img, cls in zip(images, classes)]
                    })

                # Fire and forget decoding thread
                threading.Thread(target=background_decode_and_log, args=(latents_dev, sample_classes, global_step), daemon=True).start()

    # Save checkpoint at end
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=args.ckpt_dir, target=jax_utils.unreplicate(state.params), step=global_step)
    logger.shutdown()
    print("Done")
