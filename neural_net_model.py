import logging
import os
import multiprocessing
import random
import shutil
from typing import Tuple, Callable
import time
from datetime import datetime as dt
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from mappers import Mapper


log = logging.getLogger(__name__)
SHM_PATH = "/dev/shm"
MODELS_FOLDER = "models"


class NeuralNetworkModel(nn.Module):
    def __init__(self, model_id: str, mapper: Mapper, device: str=None):
        """
        Initialize a neural network with multiple layers.
        :param mapper: maps layer creation, initialization and optimizer creation
        :param device: name of device to move to
        """
        super().__init__()
        self.model_id = model_id
        self.mapper = mapper
        self.device = device
        self.layers = nn.ModuleList(self.mapper.to_layers())
        if device:
            self.to(device)
            log.info(f"Moved model {model_id} to device {device}")
        self.optimizer: Optimizer = self.mapper.to_optimizer(self.parameters())
        self.progress = []
        self.training_data_buffer: list[Tuple[list, list | int]] = []
        self.training_buffer_size: int = self.num_params
        self.avg_cost = None
        self.avg_cost_history = []
        self.stats = None
        self.status = "Created"

    @property
    def _weights(self) -> list[Tensor]:
        """
        :return: Weight per layer None if no weight
        """
        return [p if p.ndim == 2 else None for p in self.parameters()]

    @property
    def num_params(self) -> int:
        """
        :return: Number of model parameters
        """
        return sum([p.numel() for p in self.parameters()])

    @classmethod
    def get_model_path(cls, model_id):
        return os.path.join(MODELS_FOLDER, f"model_{model_id}.pth")

    def serialize(self):
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(SHM_PATH, MODELS_FOLDER), exist_ok=True)
        model_path = self.get_model_path(self.model_id)
        model_data = {
            "layers": self.mapper.layers,
            "device": self.device,
            "state": self.state_dict(),
            "optim": self.mapper.optimizer,
            "optim_state": self.optimizer.state_dict(),
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
            "average_cost": self.avg_cost,
            "average_cost_history": self.avg_cost_history,
            "stats": self.stats,
            "status": self.status,
        }
        model_in_shm_path = os.path.join(SHM_PATH, model_path)
        log.info(f"Caching model to {model_in_shm_path}...")
        torch.save(model_data, model_in_shm_path)
        log.info(f"Model cached successfully: {model_in_shm_path}")
        p = multiprocessing.Process(target=shutil.copyfile, args=(model_in_shm_path, model_path))
        log.info(f"Offload flushing model cache {model_in_shm_path} to {model_path}...")
        p.start()

    @classmethod
    def deserialize(cls, model_id: str):
        try:
            model_path = cls.get_model_path(model_id)
            model_in_shm_path = os.path.join(SHM_PATH, model_path)
            if not os.path.exists(model_in_shm_path):
                log.info(f"Cache miss: copying from {model_path}")
                shutil.copyfile(model_path, model_in_shm_path)
            log.info(f"Retrieving model from {model_in_shm_path}...")
            data = torch.load(model_in_shm_path)
            log.info(f"Loaded model {model_id} data")
            model = cls(model_id, Mapper(data["layers"], data["optim"]))
            log.info(f"Created model {model_id}")
            model.load_state_dict(data["state"])
            log.info(f"Loaded state into model {model_id}")
            model.device = data["device"]
            if model.device:
                model.to(model.device)
                log.info(f"Moved model {model_id} to device {model.device}")
            model.optimizer.load_state_dict(data["optim_state"])
            log.info(f"Loaded optimizer for model {model_id}")
            model.progress = data["progress"]
            model.training_data_buffer = data["training_data_buffer"]
            model.training_buffer_size = model.num_params
            model.avg_cost = data["average_cost"]
            model.avg_cost_history = data["average_cost_history"]
            model.stats = data["stats"]
            model.status = data["status"]
            log.info(f"Model {model_id} is fully loaded")
            return model
        except FileNotFoundError as e:
            log.error(f"File not found error occurred: {str(e)}")
            raise KeyError(f"Model {model_id} not created yet.")

    @classmethod
    def delete(cls, model_id: str):
        try:
            model_path = cls.get_model_path(model_id)
            model_in_shm_path = os.path.join(SHM_PATH, model_path)
            os.remove(model_in_shm_path)
            if os.path.exists(model_path):
                os.remove(model_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    @torch.no_grad()
    def compute_output(self, input_data: list, target: list | int = None) -> Tuple[list, float]:
        """
        Compute activated output and optionally also cost compared to the provided target vector
        without training done.
        :param input_data: Input data
        :param target: Target data (optional)
        :return: output, cost (optional)
        """
        # output is not training
        self.eval()
        self.layers.training = False
        # forward pass
        device = next(self.parameters()).device
        log.info(f"Computing model output using device {device}")
        activations, cost = self._forward(input_data, target, device)
        # last activation  and a float cost is returned, if any
        return activations[-1].tolist(), cost.item() if cost.numel() > 0 else None

    @torch.no_grad()
    def evaluate_model(self, input_data: list, target: list, epochs=100, batch_size=None) -> float:
        """
        Evaluate cost compared to the provided target vector without training done.
        :param input_data: Input data
        :param target: Target data
        :param epochs: Number of evaluation iterations.
        :param batch_size: Batch size override for evaluation sample.
        :return: average cost
        """
        # evaluation is not training
        self.eval()
        self.layers.training = False
        # calculate sample size
        data_size = len(input_data)
        sample_size = batch_size or int(data_size / epochs)  # explicit or sample equally per epoch
        log.info(f"Evaluation sample size: {sample_size}")
        # determine device
        device = next(self.parameters()).device
        log.info(f"Evaluating model using device {device}")
        # evaluate cost each epoch and add to costs
        costs = torch.zeros(epochs)
        for epoch in range(epochs):
            # sample data
            sample_indices = torch.randint(len(input_data), (sample_size,))
            sample_input = [input_data[sample_idx] for sample_idx in sample_indices]
            sample_target = [target[sample_idx] for sample_idx in sample_indices]
            # forward pass
            _, sample_cost = self._forward(sample_input, sample_target, device)
            # record sample cost
            costs[epoch] = sample_cost
            # Log each
            log.info(f"Model {self.model_id}: Evaluation Epoch {epoch + 1}, Cost: {sample_cost:.4f}")
        # calculate and return average sample cost
        avg_cost = costs.mean().item()
        return avg_cost

    @torch.no_grad()
    def generate_tokens(self, input_context: list, block_size: int, max_new_tokens: int) -> list:
        # generating is not training
        self.eval()
        self.layers.training = False
        # initialize context
        device = next(self.parameters()).device
        log.info(f"Generating tokens using device {device}")
        context = torch.tensor(input_context, dtype=torch.long, device=device)
        # generate up to max new tokens
        for sample_idx in range(max_new_tokens):
            # crop context to the last block size tokens
            cropped_context = context[:, -block_size:]
            # Predict next token
            activations, _ = self(cropped_context)
            probs = activations[-1]
            next_idx = torch.multinomial(probs, num_samples=1)
            # Append next token in for next prediction
            context = torch.cat((context, next_idx), dim=1)
        # extract and return tokens
        tokens = context[0].tolist()
        return tokens

    def _forward(self, input_data: list, target: list | int, device=None) -> Tuple[list[Tensor], Tensor]:
        input_tensor = torch.tensor(input_data, device=device)
        return self(input_tensor, target)

    def forward(self, input_tensor: Tensor, target: list | int=None) -> Tuple[list[Tensor], Tensor]:
        forwarded_tensors = []
        forwarded_tensor = input_tensor
        previous_tensor = input_tensor
        for layer in self.layers:
            previous_tensor = forwarded_tensor
            forwarded_tensor = layer(previous_tensor)
            forwarded_tensors.append(forwarded_tensor)

        if target is None or target == []:
            cost = torch.empty(0)
        elif isinstance(self.layers[-1], nn.Softmax):
            label_tensor = torch.tensor(target, dtype=torch.int64, device=input_tensor.device)
            if previous_tensor.ndim > 2 and label_tensor.ndim > 1: # e.g. transformer cost
                logits = previous_tensor.view(-1, previous_tensor.size(-1))
                label_tensor = label_tensor.view(-1)
            else:
                logits = previous_tensor
            cost = nn.functional.cross_entropy(logits, label_tensor)
        else:
            target_tensor = torch.tensor(target, device=input_tensor.device)
            cost = nn.functional.mse_loss(forwarded_tensor, target_tensor)

        return forwarded_tensors, cost

    def train_model(self, input_data: list, target: list, epochs=100, batch_size=None):
        """
        Train the neural network using the provided training data.
        :param input_data: input data
        :param target: target data
        :param epochs: Number of training iterations.
        :param batch_size: Batch size override for training sample.
        """
        # Combine incoming training data with buffered data
        self.training_data_buffer.extend(zip(input_data, target))

        # Check if buffer size is sufficient
        current_buffer_size = len(self.training_data_buffer)
        if current_buffer_size < self.training_buffer_size:
            log.info(f"Model {self.model_id}: Insufficient training data. "
                  f"Current buffer size: {current_buffer_size}, "
                  f"required: {self.training_buffer_size}")
            self.serialize() # serialize model with partial training data for next time
            return

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        # Calculate sample size
        training_sample_size = batch_size or int(len(training_data) / epochs)  # explicit or sample equally per epoch
        log.info(f"Training sample size: {training_sample_size}")

        # Determine device
        device = next(self.parameters()).device
        log.info(f"Training model using device {device}")

        # Reset model for training prep and save
        self.progress = []
        self.stats = None
        self.status = "Training"
        self.serialize()
        last_serialized = time.time()
        activations = None
        self.train()
        self.layers.training = True

        # Start training
        for epoch in range(epochs):
            random_indices = torch.randint(0, len(training_data), (training_sample_size,))
            training_sample = [training_data[i] for i in random_indices]

            # copy weights for later update ratio calc
            prev_weights: list[Tensor] = [None if w is None else w.clone().detach() for w in self._weights]
            # organize training data for forward pass
            input_sample = [inpt for inpt, _ in training_sample]
            target_sample = [tgt for _, tgt in training_sample]
            # calculate cost
            activations, cost = self._forward(input_sample, target_sample, device)
            # check if training taking long
            long_training = time.time() - last_serialized >= 10
            # clear gradients
            self.optimizer.zero_grad()
            # on last epoch or for long training intervals
            # retain final activation gradients to collect stats
            if epoch + 1 == epochs or long_training:
                for a in activations:
                    a.retain_grad()
            # back propagate to populate gradients
            cost.backward()
            # optimize parameters
            self.optimizer.step()

            # Record progress
            progress_dt, progress_cost = dt.now().isoformat(), cost.item()
            if epoch % max(1, epochs // 100) == 0: # only 100 progress points or less stored
                with torch.no_grad():
                    self.progress.append({
                        "dt": progress_dt,
                        "epoch": epoch + 1,
                        "cost": progress_cost,
                        "weight_upd_ratio": [
                            None if w is None or pw is None else ((w - pw).data.std() / (w.data.std() + 1e-8)).item()
                            for pw, w in zip(prev_weights, self._weights)
                        ],
                    })
            # Log each
            log.info(f"Model {self.model_id}: Training Epoch {epoch + 1}, Cost: {progress_cost:.4f}")

            # Serialize model while long training intervals
            if long_training: # pragma: no cover
                self._record_training_overall_progress(activations)
                self.serialize()
                last_serialized = time.time()

        # Mark training finished
        self.status = "Trained"
        # Log training is done
        log.info(f"Model {self.model_id}: Done training for {epochs} epochs.")
        # Serialize model after training
        self._record_training_overall_progress(activations)
        self.serialize()

    @torch.no_grad()
    def _record_training_overall_progress(self, activations):
        # Calculate current average progress cost
        progress_cost = [progress["cost"] for progress in self.progress]
        avg_progress_cost = sum(progress_cost) / len(self.progress)
        # Update overall average cost
        self.avg_cost = ((self.avg_cost or avg_progress_cost) + avg_progress_cost) / 2.0
        self.avg_cost_history.append(self.avg_cost)
        if len(self.avg_cost_history) > 100: #
            self.avg_cost_history.pop(random.randint(1, 98))
        # Update stats
        hist_f: Callable[[torch.return_types.histogram], Tuple[list, list]] = (
            lambda h: (h.bin_edges[:-1].tolist(), h.hist.tolist()))
        act_hist = [hist_f(torch.histogram(a.cpu(), density=True)) for a in activations]
        act_grad_hist = [([], []) if a.grad is None else hist_f(torch.histogram(a.grad.cpu(), density=True))
                         for a in activations]
        weight_grad_hist = [([], []) if w is None else hist_f(torch.histogram(w.grad.cpu(), density=True))
                            for w in self._weights]
        algos = [l.__class__.__name__.lower() for l in self.layers]
        self.stats = {
            "layers": [{
                "algo": algo,
                "activation": {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "saturated": (
                        (torch.norm(a, dim=-1) > 5.0) if algo == "embedding" else
                        (a.abs() > 3.0) if algo == "batchnorm1d" else
                        (a.abs() > 0.97) if algo in ["tanh", "sigmoid"] else
                        (a <= 0) if algo == "relu" else
                        (a.max(dim=-1).values > 0.97) if algo == "softmax" else
                        (a.abs() > 5.0) # if l.algo == "linear" or etc.
                    ).float().mean().item(),
                    "histogram": {"x": ahx, "y": ahy},
                },
                "gradient": {
                    "mean": a.grad.mean().item(),
                    "std": a.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                } if a.grad is not None else None,
            } for algo, a, (ahx, ahy), (ghx, ghy) in zip(algos, activations, act_hist, act_grad_hist)],
            "weights": [{
                "shape": str(tuple(w.shape)),
                "data": {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                },
                "gradient": {
                    "mean": w.grad.mean().item(),
                    "std": w.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                },
            } if w is not None else None for w, (ghx, ghy) in zip(self._weights, weight_grad_hist)],
        }
        # Log training progress
        log.info(f"Model {self.model_id} - Cost: {avg_progress_cost:.4f} Overall Cost: {self.avg_cost:.4f}")
