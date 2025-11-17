"""
Meta-Learning Training for Learned Optimizers

Implements MAML-style meta-learning to train optimizer memory modules
on a distribution of tasks.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple
from tqdm import tqdm


class MetaLearner:
    """
    Meta-learner for training optimizer memory modules.

    Uses MAML-style meta-learning:
    1. Sample tasks from distribution
    2. For each task, run K optimization steps using learned optimizer
    3. Evaluate on validation set
    4. Backprop through unrolled optimization to update memory modules
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        optimizer_class: type,
        optimizer_kwargs: dict,
        device: str = 'cpu',
    ):
        """
        Args:
            model_fn: Function that creates a fresh model instance
            optimizer_class: Optimizer class (e.g., NestedDeepMomentumGD)
            optimizer_kwargs: Kwargs for optimizer
            device: Device to run on
        """
        self.model_fn = model_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.device = device

    def meta_train_step(
        self,
        task_batch: List[Callable],
        num_inner_steps: int = 5,
        inner_lr: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Perform one meta-training step.

        Args:
            task_batch: Batch of task generator functions
            num_inner_steps: Number of inner loop optimization steps per task
            inner_lr: Override learning rate for inner loop

        Returns:
            (pre_loss, post_loss): Losses before and after adaptation
        """
        total_pre_loss = 0.0
        total_post_loss = 0.0

        for task_fn in task_batch:
            # Create fresh model for this task
            model = self.model_fn().to(self.device)

            # Create optimizer (with shared memory modules if using NestedDMGD)
            if inner_lr is not None:
                opt_kwargs = {**self.optimizer_kwargs, 'lr': inner_lr}
            else:
                opt_kwargs = self.optimizer_kwargs

            optimizer = self.optimizer_class(model.parameters(), **opt_kwargs)

            # Generate task data
            X_train, y_train = task_fn()
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            # Evaluate before adaptation
            model.eval()
            with torch.no_grad():
                pre_loss = self._compute_loss(model, X_train, y_train)
                total_pre_loss += pre_loss.item()

            # Inner loop: Adapt model to task
            model.train()
            for _ in range(num_inner_steps):
                optimizer.zero_grad()

                def closure():
                    optimizer.zero_grad()
                    loss = self._compute_loss(model, X_train, y_train)
                    loss.backward(create_graph=True)  # create_graph=True for meta-learning
                    return loss

                optimizer.step(closure)

            # Evaluate after adaptation (this is the meta-loss)
            X_val, y_val = task_fn()  # New batch for validation
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

            post_loss = self._compute_loss(model, X_val, y_val)
            total_post_loss += post_loss.item()

            # Backprop through adaptation to update memory modules
            if hasattr(optimizer, 'meta_step'):
                optimizer.meta_step(post_loss)

        avg_pre_loss = total_pre_loss / len(task_batch)
        avg_post_loss = total_post_loss / len(task_batch)

        return avg_pre_loss, avg_post_loss

    def meta_train(
        self,
        task_distribution: List[Callable],
        num_iterations: int = 1000,
        meta_batch_size: int = 4,
        num_inner_steps: int = 5,
        eval_every: int = 100,
    ):
        """
        Run full meta-training.

        Args:
            task_distribution: List of task generator functions
            num_iterations: Number of meta-training iterations
            meta_batch_size: Number of tasks per meta-batch
            num_inner_steps: Number of inner optimization steps
            eval_every: Evaluate and print every N iterations
        """
        print(f"Meta-training for {num_iterations} iterations...")
        print(f"Task distribution size: {len(task_distribution)}")
        print(f"Meta batch size: {meta_batch_size}")
        print(f"Inner steps: {num_inner_steps}")

        for iteration in tqdm(range(num_iterations)):
            # Sample batch of tasks
            task_batch = [
                task_distribution[torch.randint(0, len(task_distribution), (1,)).item()]
                for _ in range(meta_batch_size)
            ]

            # Meta-training step
            pre_loss, post_loss = self.meta_train_step(
                task_batch,
                num_inner_steps=num_inner_steps,
            )

            # Log progress
            if (iteration + 1) % eval_every == 0:
                improvement = pre_loss - post_loss
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  Pre-adaptation loss:  {pre_loss:.4f}")
                print(f"  Post-adaptation loss: {post_loss:.4f}")
                print(f"  Improvement: {improvement:.4f}")

    def _compute_loss(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        output = model(X)

        # Detect task type from shapes
        if y.dtype == torch.long or (y.dtype == torch.float and y.ndim == 1):
            # Classification
            loss = nn.functional.cross_entropy(output, y)
        else:
            # Regression
            loss = nn.functional.mse_loss(output, y)

        return loss


def pretrain_optimizer(
    optimizer_class: type,
    optimizer_kwargs: dict,
    task_distribution: List[Callable],
    model_fn: Callable[[], nn.Module],
    num_iterations: int = 1000,
    meta_batch_size: int = 4,
    num_inner_steps: int = 5,
    device: str = 'cpu',
) -> nn.ModuleDict:
    """
    Pretrain an optimizer's memory modules on a task distribution.

    This is the main entry point for meta-learning.

    Args:
        optimizer_class: Optimizer class to train
        optimizer_kwargs: Optimizer hyperparameters
        task_distribution: List of task generators
        model_fn: Function that creates model instances
        num_iterations: Number of meta-training iterations
        meta_batch_size: Tasks per meta-batch
        num_inner_steps: Inner loop steps
        device: Device to use

    Returns:
        Trained memory modules (as ModuleDict)
    """
    meta_learner = MetaLearner(
        model_fn=model_fn,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )

    meta_learner.meta_train(
        task_distribution=task_distribution,
        num_iterations=num_iterations,
        meta_batch_size=meta_batch_size,
        num_inner_steps=num_inner_steps,
    )

    # Extract trained memory modules
    model = model_fn().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    if hasattr(optimizer, 'get_memory_modules'):
        memory_modules = nn.ModuleDict({
            f'memory_{i}': module
            for i, module in enumerate(optimizer.get_memory_modules())
        })
        return memory_modules
    else:
        return nn.ModuleDict()


# Example task distributions

def create_regression_tasks(
    num_tasks: int = 100,
    input_dim: int = 20,
    output_dim: int = 1,
) -> List[Callable]:
    """Create distribution of regression tasks."""
    tasks = []

    for _ in range(num_tasks):
        # Random linear regression with noise
        true_w = torch.randn(input_dim, output_dim)
        true_b = torch.randn(output_dim)
        noise_std = 0.1 * torch.rand(1).item()

        def make_task(w=true_w, b=true_b, std=noise_std):
            def task():
                X = torch.randn(32, input_dim)
                y = X @ w + b + std * torch.randn(32, output_dim)
                return X, y
            return task

        tasks.append(make_task())

    return tasks


def create_sinusoid_tasks(num_tasks: int = 100) -> List[Callable]:
    """Create distribution of sinusoid regression tasks (classic MAML)."""
    tasks = []

    for _ in range(num_tasks):
        # Random amplitude and phase
        amplitude = torch.rand(1).item() * 5.0 + 0.1
        phase = torch.rand(1).item() * torch.pi

        def make_task(amp=amplitude, ph=phase):
            def task():
                X = torch.rand(32, 1) * 10 - 5  # [-5, 5]
                y = amp * torch.sin(X + ph)
                return X, y
            return task

        tasks.append(make_task())

    return tasks
