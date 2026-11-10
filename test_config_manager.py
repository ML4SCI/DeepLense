"""Tests for config_manager.py — Configuration Manager for DeepLense."""

import json
import os
import tempfile
import pytest

from config_manager import (
    ConfigManager,
    DEFAULT_CLASSIFICATION_CONFIG,
    DEFAULT_REGRESSION_CONFIG,
)


class TestConfigManagerDefaults:
    """Test creating configs from defaults."""
    
    def test_classification_defaults(self):
        config = ConfigManager.from_defaults("classification")
        assert config.get("model.architecture") == "resnet18"
        assert config.get("model.num_classes") == 3
        assert config.get("training.epochs") == 50
        assert config.get("training.learning_rate") == 0.001
    
    def test_regression_defaults(self):
        config = ConfigManager.from_defaults("regression")
        assert config.get("model.num_outputs") == 1
        assert config.get("training.epochs") == 100
        assert config.get("training.scheduler") == "reduce_on_plateau"
    
    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            ConfigManager.from_defaults("invalid")
    
    def test_defaults_are_independent_copies(self):
        c1 = ConfigManager.from_defaults("classification")
        c2 = ConfigManager.from_defaults("classification")
        c1.set("training.epochs", 999)
        assert c2.get("training.epochs") == 50  # unchanged


class TestConfigManagerGetSet:
    """Test get and set with dot notation."""
    
    def test_get_nested_value(self):
        config = ConfigManager.from_defaults("classification")
        assert config.get("training.scheduler_params.T_max") == 50
    
    def test_get_missing_key_returns_default(self):
        config = ConfigManager.from_defaults("classification")
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", 42) == 42
    
    def test_set_existing_value(self):
        config = ConfigManager.from_defaults("classification")
        config.set("training.learning_rate", 0.01)
        assert config.get("training.learning_rate") == 0.01
    
    def test_set_new_nested_value(self):
        config = ConfigManager.from_defaults("classification")
        config.set("custom.new.param", "hello")
        assert config.get("custom.new.param") == "hello"
    
    def test_bracket_access(self):
        config = ConfigManager.from_defaults("classification")
        assert config["training.epochs"] == 50
        config["training.epochs"] = 200
        assert config["training.epochs"] == 200


class TestConfigManagerFileIO:
    """Test saving and loading configuration files."""
    
    def test_save_and_load_json(self):
        config = ConfigManager.from_defaults("classification")
        config.set("training.epochs", 75)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        
        try:
            config.save(filepath)
            loaded = ConfigManager.from_file(filepath, merge_defaults=False)
            assert loaded.get("training.epochs") == 75
            assert loaded.get("model.architecture") == "resnet18"
        finally:
            os.unlink(filepath)
    
    def test_load_with_merge_defaults(self):
        partial_config = {"training": {"epochs": 200}}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(partial_config, f)
            filepath = f.name
        
        try:
            config = ConfigManager.from_file(filepath, merge_defaults=True)
            assert config.get("training.epochs") == 200  # overridden
            assert config.get("training.learning_rate") == 0.001  # from defaults
            assert config.get("model.architecture") == "resnet18"  # from defaults
        finally:
            os.unlink(filepath)
    
    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager.from_file("nonexistent_file.json")
    
    def test_load_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                ConfigManager.from_file(filepath)
        finally:
            os.unlink(filepath)


class TestConfigManagerValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        config = ConfigManager.from_defaults("classification")
        errors = config.validate()
        assert len(errors) == 0
    
    def test_invalid_learning_rate(self):
        config = ConfigManager.from_defaults("classification")
        config.set("training.learning_rate", -0.01)
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)
    
    def test_invalid_epochs(self):
        config = ConfigManager.from_defaults("classification")
        config.set("training.epochs", 0)
        errors = config.validate()
        assert any("epochs" in e for e in errors)
    
    def test_invalid_data_splits(self):
        config = ConfigManager.from_defaults("classification")
        config.set("data.train_split", 0.5)
        config.set("data.val_split", 0.5)
        config.set("data.test_split", 0.5)
        errors = config.validate()
        assert any("splits" in e.lower() for e in errors)
    
    def test_invalid_batch_size(self):
        config = ConfigManager.from_defaults("classification")
        config.set("training.batch_size", -1)
        errors = config.validate()
        assert any("batch_size" in e for e in errors)
    
    def test_invalid_seed(self):
        config = ConfigManager.from_defaults("classification")
        config.set("seed", "not_a_number")
        errors = config.validate()
        assert any("seed" in e for e in errors)


class TestConfigManagerDiff:
    """Test configuration comparison."""
    
    def test_identical_configs_no_diff(self):
        c1 = ConfigManager.from_defaults("classification")
        c2 = ConfigManager.from_defaults("classification")
        assert len(c1.diff(c2)) == 0
    
    def test_diff_detects_changes(self):
        c1 = ConfigManager.from_defaults("classification")
        c2 = ConfigManager.from_defaults("classification")
        c2.set("training.epochs", 999)
        diffs = c1.diff(c2)
        assert "training.epochs" in diffs
        assert diffs["training.epochs"] == (50, 999)
    
    def test_diff_detects_new_keys(self):
        c1 = ConfigManager.from_defaults("classification")
        c2 = ConfigManager.from_defaults("classification")
        c2.set("custom.param", "value")
        diffs = c1.diff(c2)
        # The diff may nest under "custom" since it's a new top-level key
        has_custom = any("custom" in k for k in diffs)
        assert has_custom


class TestConfigManagerFromDict:
    """Test creating configs from dictionaries."""
    
    def test_from_dict(self):
        config = ConfigManager.from_dict({"model": {"name": "test"}})
        assert config.get("model.name") == "test"
    
    def test_to_dict(self):
        config = ConfigManager.from_defaults("classification")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["training"]["epochs"] == 50
    
    def test_to_dict_is_copy(self):
        config = ConfigManager.from_defaults("classification")
        d = config.to_dict()
        d["training"]["epochs"] = 999
        assert config.get("training.epochs") == 50  # unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
