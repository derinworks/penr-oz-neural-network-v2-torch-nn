import math
import os.path
import time
import unittest

import torch
from parameterized import parameterized
import numpy as np
import torch.nn as nn
from neural_net_model import NeuralNetworkModel, SHM_PATH
from mappers import Mapper
import neural_net_layers as nnl


class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 9}, "xavier_uniform": {}}, {"relu": {}}],
         {"adam": {"lr": 0.1}}, "cpu",
         [nn.Linear,nn.ReLU], [(9,9),(9,)], 90),
        ([{"linear": {"in_features": 18, "out_features": 9}, "xavier_uniform": {}}, {"softmax": {"dim": -1}}],
         {"adamw": {"lr": 0.1}}, None,
         [nn.Linear,nn.Softmax], [(9,18),(9,)], 171),
        ([{"linear": {"in_features": 9, "out_features": 18, "bias": False}, "kaiming_uniform": {}}, {"sigmoid": {}}], 
         {"sgd": {"lr": 0.1}}, "cpu",
         [nn.Linear,nn.Sigmoid], [(18,9)], 162),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"tanh": {}}], {"sgd": {"lr": 0.1}}, None,
         [nn.Linear,nn.Tanh] * 2, [(8,4),(8,), (16,8),(16,)], 184),
        ([{"linear": {"in_features": 3, "out_features": 3, "bias": False}}, {"relu": {}},
          {"linear": {"in_features": 3, "out_features": 3}}, {"tanh": {}},
          {"linear": {"in_features": 3, "out_features": 3, "bias": False}, "xavier_uniform": {}}, {"softmax": {"dim": -1}}
          ], {"sgd": {"lr": 0.1}}, None,
         [nn.Linear,nn.ReLU, nn.Linear,nn.Tanh, nn.Linear,nn.Softmax], [(3,3), (3,3),(3,), (3,3)], 30),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 20}}, {"tanh": {}},
          {"linear": {"in_features": 20, "out_features": 18, "bias": False}}, {"softmax": {"dim": -1}},
          ], {"sgd": {"lr": 0.1}}, None,
         [nn.Embedding,nn.Flatten, nn.Linear,nn.Tanh, nn.Linear,nn.Softmax], [(18,2),(20,6),(20,), (18,20)], 536),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 20}}, {"batchnorm1d": {"num_features": 20}}, {"tanh": {}},
          {"linear": {"in_features": 20, "out_features": 18, "bias": False}, "confidence": 0.1}, {"softmax": {"dim": -1}},
          ], {"sgd": {"lr": 0.1}}, None,
         [nn.Embedding,nn.Flatten, nn.Linear,nn.BatchNorm1d,nn.Tanh, nn.Linear,nn.Softmax],
         [(18,2),(20,6),(20,),(20,),(20,),(18,20)], 576),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 10}}, {"tanh": {}},
          {"linear": {"in_features": 10, "out_features": 18}}, {"dropout": {"p": 0.1}},{"softmax": {"dim": -1}},
         ], {"sgd": {"lr": 0.1}}, None,
         [nn.Embedding,nn.Flatten, nn.Linear,nn.Tanh, nn.Linear,nn.Dropout,nn.Softmax],
         [(18,2),(10,6),(10,),(18,10),(18,)], 304),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}], {"adam": {"lr": 3e-4}}, None,
         [nnl.Summation],
         [(27, 4), (8, 4)], 140),
        ([{"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                          {"linear": {"in_features": 4, "out_features": 12},
                           "normal": {"std": 0.2}, "zeros": {}},
                          {"attention": {"num_heads": 2}},
                          {"linear": {"in_features": 4, "out_features": 4},
                           "normal": {"std": 0.2}, "zeros": {}},
                          ]}
          ], {"adamw": {"lr": 3e-4}}, None,
         [nn.Sequential],
         [(4,), (12, 4), (12,), (4, 4), (4,)], 84),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}, {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
        [{"layernorm": {"normalized_shape": 4, "bias": False}},
         {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
         {"softmaxlast": {"dim": -1}}], {"adamw": {"lr": 3e-4}}, "cpu",
         [nnl.Summation,nn.Dropout] + [nnl.ResidualConnection] * 2 + [nn.LayerNorm,nn.Linear,nnl.SoftmaxOnLast],
         [(27, 4), (8, 4)] + [(4,), (12, 4), (4, 4), (4,), (16, 4), (4, 16)] * 2 + [(4,), (27, 4)], 652),
    ])
    def test_model_init(self, layers: list[dict], optimizer: dict, device: str,
                        expected_layers: list[nn.Module], expected_shapes: list[list[tuple]], expected_buffer_size: int):

        model = NeuralNetworkModel("test", Mapper(layers, optimizer), device)

        self.assertEqual("test", model.model_id)
        self.assertEqual(device, model.device)
        if device:
            self.assertEqual(torch.device(device), next(model.parameters()).device)
        self.assertListEqual(expected_layers, [l.__class__ for l in model.layers])
        self.assertListEqual(expected_shapes, [tuple(p.shape) for p in model.parameters()])
        self.assertTrue(model.optimizer.__class__.__name__.lower() in optimizer.keys())
        self.assertEqual(0, len(model.progress))
        self.assertEqual(expected_buffer_size, model.training_buffer_size)
        self.assertEqual(expected_buffer_size, model.num_params)
        self.assertIsNone(model.avg_cost)
        self.assertEqual(0, len(model.avg_cost_history))
        self.assertIsNone(model.stats)
        self.assertEqual("Created", model.status)

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 9}}, {"sigmoid": {}}] * 2, [0.5] * 9, None),
        ([{"linear": {"in_features": 9, "out_features": 9}}, {"softmax": {"dim": 0}}], [1.0] + [0.0] * 8, 4),
        ([{"linear": {"in_features": 18, "out_features": 9}}, {"relu": {}},
          {"linear": {"in_features": 9, "out_features": 3}}, {"softmax": {"dim": 0}}], [1.0] + [0.0] * 17, None),
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, [0.5] * 9, [0.5] * 9),
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, [[0.5] * 9] * 2, [[0.5] * 9] * 2),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 0}}], [0.5] * 4, 13),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 1}}], [[0.5] * 4] * 2, [13] * 2),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}], [[0, 5, 8],[1, 3, 7]], [2, 4]),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}, {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}], [[1,12,21,5,8,10,5,17]] * 5, [[12,21,5,8,10,5,17,21]] * 5),
    ])
    def test_compute_output(self, layers: list[dict], input_data: list, target: list | int | None):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        output, cost = model.compute_output(input_data, target)
        in_shape = np.shape(input_data)
        out_shape = np.shape(output)

        self.assertEqual(len(in_shape), len(out_shape))
        if len(out_shape) > 1: # same batch size?
            self.assertEqual(in_shape[0], out_shape[0])
        self.assertTrue(target is None or cost is not None)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2,
         [[0.5] * 9] * 2, [[0.5] * 9] * 2, 1, None),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 1}}],
         [[0.5] * 4] * 8, [13] * 8, 2, 2),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8],[1, 3, 7]] * 10, [2, 4] * 10, 3, 3),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[1,12,21,5,8,10,5,17]] * 24, [[12,21,5,8,10,5,17,21]] * 24, 4, 3),
    ])
    def test_evaluate(self, layers: list[dict], input_data: list, target: list,
                      epochs: int, batch_size: int | None):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        cost = model.evaluate_model(input_data, target, epochs, batch_size)

        self.assertIsNotNone(cost)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8]], 3, 3),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[0]], 8, 10),
    ])
    def test_generate_tokens(self, layers: list[dict], input_context: list, block_size: int, max_new_tokens: int):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        tokens = model.generate_tokens(input_context, block_size, max_new_tokens)

        self.assertIsNotNone(tokens)
        self.assertGreaterEqual(len(tokens), block_size)
        self.assertLessEqual(len(tokens), len(input_context[0]) + max_new_tokens)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 9}}, {"softmax": {"dim": 1}}], {"sgd": {"lr": 0.1}}, None,
         [[1.0] + [0.0] * 8] * 90, [4] * 90, 3, None),
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, {"adam": {"lr": 0.1}}, None,
         [[0.5] * 9] * 702, [[0.5] * 9] * 702, 4, 20),
        ([{"linear": {"in_features": 9, "out_features": 18, "bias": False}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, {"adamw": {"lr": 0.1}}, "cpu",
         [[0.5] * 9] * 666, [[0.5] * 9] * 666, 5, 40),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 1}}], {"sgd": {"lr": 0.1}}, "cpu",
         [[0.5] * 4] * 184, [13] * 184, 5, 32),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16, "bias": False}}, {"softmax": {"dim": 1}}],
         {"sgd": {"lr": 0.01}}, None,
         [[0.5] * 4] * 168, [13] * 168, 3, 48),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 10}}, {"batchnorm1d": {"num_features": 10}}, {"tanh": {}},
          {"linear": {"in_features": 10, "out_features": 18}}, {"dropout": {"p": 0.1}}, {"softmax": {"dim": 1}}],
         {"adamw": {"lr": 0.1, "betas": [0.99, 0.9999], "eps": 1e-9, "weight_decay": 1e-1}}, "cpu",
         [[0, 5, 8],[1, 3, 7]] * 162, [2, 4] * 162, 5, 32),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}, {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12},
                              "normal": {"mean": 0.0, "std": 0.2}, "zeros": {}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4},
                              "normal": {"mean": 0.0, "std": 0.2 / math.sqrt(2)}, "zeros": {}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16},
                              "normal": {"mean": 0.0, "std": 0.2}, "zeros": {}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4},
                              "normal": {"mean": 0.0, "std": 0.2 / math.sqrt(2)}, "zeros": {}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": 3e-4}}, "cpu",
         [[1,12,21,5,8,10,5,17]] * 768, [[12,21,5,8,10,5,17,21]] * 768, 5, 64)
    ])
    def test_train(self, layers: list[dict], optimizer: dict, device: str, input_data: list, target: list,
                   epochs: int, batch_size: int | None):
        # clean up any persisted previous test model
        NeuralNetworkModel.delete("test")

        # create model
        model = NeuralNetworkModel("test", Mapper(layers, optimizer), device)

        # record initial conditions
        initial_params = [p.tolist() for p in model.parameters()]
        _, initial_cost = model.compute_output(input_data, target)
        lr: float = model.optimizer.param_groups[0]["lr"]

        # Add average cost history to test cap at 100
        model.avg_cost_history = [1.0] * 100

        # make sure test data is good for training
        self.assertEqual(len(input_data), len(target))
        self.assertGreaterEqual(len(input_data), model.training_buffer_size)

        model.train_model(input_data, target, epochs=epochs, batch_size=batch_size)

        # record updated
        updated_params = [p.tolist() for p in model.parameters()]
        updated_optim_params =[p.tolist() for p in model.optimizer.param_groups[0]["params"]]

        # Check that the model data is still valid
        self.assertEqual(len(updated_params), len(initial_params))
        for u, i in zip(updated_params, initial_params):
            self.assertEqual(np.shape(u), np.shape(i))

        # Ensure training progress
        for u, i in zip(updated_params, initial_params):
            self.assertFalse(np.allclose(u, i))
        self.assertEqual(len(model.progress), epochs)
        self.assertNotEqual(model.progress[-1]["cost"], initial_cost)
        self.assertEqual(sum([p["cost"] for p in model.progress]) / len(model.progress), model.avg_cost)
        self.assertEqual(len(model.avg_cost_history), 100)
        self.assertEqual(model.avg_cost_history[0], 1.0)
        self.assertEqual(model.avg_cost_history[-1], model.avg_cost)
        self.assertEqual(len(model.training_data_buffer), 0)
        self.assertIsNotNone(model.stats)
        self.assertEqual("Trained", model.status)
        self.assertTrue(model.layers.training)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        # record persisted
        persisted_params = [p.tolist() for p in persisted_model.parameters()]
        persisted_lr: float = persisted_model.optimizer.param_groups[0]["lr"]
        persisted_optim_params = [p.tolist() for p in persisted_model.optimizer.param_groups[0]["params"]]

        # Verify model correctly deserialized
        self.assertEqual(len(persisted_params), len(updated_params))
        for p, u in zip(persisted_params, updated_params):
            self.assertEqual(np.shape(p), np.shape(u))
            np.testing.assert_allclose(p, u)
        self.assertEqual(persisted_model.optimizer.__class__, model.optimizer.__class__)
        self.assertEqual(persisted_lr, lr)
        for p, u in zip(persisted_optim_params, updated_optim_params):
            self.assertEqual(np.shape(p), np.shape(u))
            np.testing.assert_allclose(p, u, rtol=1e-5, atol=1e-8)
        self.assertEqual(len(persisted_model.progress), len(model.progress))
        self.assertEqual(len(persisted_model.training_data_buffer), 0)
        self.assertEqual(persisted_model.avg_cost, model.avg_cost)
        self.assertEqual(persisted_model.avg_cost_history, model.avg_cost_history)
        self.assertEqual(persisted_model.stats, model.stats)
        self.assertEqual(persisted_model.status, model.status)
        self.assertEqual(persisted_model.device, model.device)
        if persisted_model.device:
            self.assertEqual(torch.device(persisted_model.device), next(persisted_model.parameters()).device)


    def test_train_with_insufficient_data(self):
        # Test that training does not proceed when data is less than the buffer size
        sz = 9
        input_data = [0.5] * sz
        target = 5
        model = NeuralNetworkModel("test", Mapper(
            [{"linear": {"in_features": sz, "out_features": sz}}, {"softmax": {"dim": -1}}],
            {"sgd": {}}))

        training_data_size = 2

        # ensure it is insufficient data
        self.assertLess(training_data_size, model.training_buffer_size)

        # try training with insufficient data
        model.train_model([input_data] * training_data_size, [target] * training_data_size,  epochs=1)

        # Ensure no training progress and buffering
        self.assertEqual(len(model.progress), 0)
        self.assertEqual(len(model.training_data_buffer), training_data_size)

        # Deserialize and check if recorded training buffer
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        self.assertEqual(len(persisted_model.training_data_buffer), len(model.training_data_buffer))

        # now try adding enough to train with sufficient data
        training_data_size = persisted_model.training_buffer_size - training_data_size
        persisted_model.train_model([input_data] * training_data_size, [target] * training_data_size,  epochs=1)

        # Ensure training progress and buffer flushed
        self.assertEqual(len(persisted_model.progress), 1)
        self.assertEqual(len(persisted_model.training_data_buffer), 0)

    def test_unsupported_layer(self):
        with self.assertRaises(ValueError) as context:
            NeuralNetworkModel("test", Mapper([{"unknown": {}}], {"sgd": {}}))

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported layer: {'unknown': {}}")

    def test_unsupported_optimizer(self):
        with self.assertRaises(ValueError) as context:
            NeuralNetworkModel("test", Mapper([{"relu": {}}], {"unknown": {}}))

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported optimizer: {'unknown': {}}")

    def test_invalid_model_deserialization(self):
        # Test that deserializing a nonexistent model raises a KeyError
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("nonexistent_model")

    def test_delete(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(SHM_PATH, model_path)

        self.assertTrue(os.path.exists(model_in_shm_path))
        time.sleep(1) # wait a bit for cache to flush to disk
        self.assertTrue(os.path.exists(model_path))

        NeuralNetworkModel.delete("test")
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("test")

    def test_invalid_delete(self):
        # No error raised for failing to delete
        NeuralNetworkModel.delete("nonexistent")

    def test_cache_miss(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(SHM_PATH, model_path)

        self.assertTrue(os.path.exists(model_in_shm_path))
        time.sleep(1) # wait a bit for cache to flush to disk
        self.assertTrue(os.path.exists(model_path))

        os.remove(model_in_shm_path)

        self.assertFalse(os.path.exists(model_in_shm_path))

        model = NeuralNetworkModel.deserialize("test")

        self.assertIsNotNone(model)
        self.assertTrue(os.path.exists(model_in_shm_path))


if __name__ == '__main__':
    unittest.main()
