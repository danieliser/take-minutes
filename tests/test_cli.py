"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from minutes.cli import main
from minutes.models import ExtractionResult


class TestProcessCommand:
    """Test the process command."""

    def test_process_valid_file(self, tmp_path):
        """Process a valid text file with mocked backend."""
        with patch('minutes.cli_process.get_backend') as mock_get_backend:
            # Create mock backend
            mock_backend = Mock()
            extraction_result = ExtractionResult(
                decisions=[],
                ideas=[],
                questions=[],
                action_items=[],
                concepts=[],
                terms=[],
                tldr="Test TLDR"
            )
            mock_backend.generate.return_value = json.dumps(extraction_result.model_dump())
            mock_get_backend.return_value = (mock_backend, "mock")

            # Create test input file
            input_file = tmp_path / "test.txt"
            input_file.write_text("Meeting notes here")

            # Run CLI
            runner = CliRunner()
            result = runner.invoke(main, ['process', str(input_file), '--output', str(tmp_path)])

            # Verify success
            assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            assert "✓" in result.output or "Success" in result.output.lower() or str(tmp_path) in result.output

    def test_process_nonexistent_file(self, tmp_path):
        """Process nonexistent file returns exit code 1."""
        runner = CliRunner()
        nonexistent = str(tmp_path / "nonexistent.txt")
        result = runner.invoke(main, ['process', nonexistent, '--output', str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_process_with_no_dedup_flag(self, tmp_path):
        """Process with --no-dedup flag."""
        with patch('minutes.cli_process.get_backend') as mock_get_backend:
            mock_backend = Mock()
            extraction_result = ExtractionResult(
                decisions=[],
                ideas=[],
                questions=[],
                action_items=[],
                concepts=[],
                terms=[],
                tldr="Test TLDR"
            )
            mock_backend.generate.return_value = json.dumps(extraction_result.model_dump())
            mock_get_backend.return_value = (mock_backend, "mock")

            input_file = tmp_path / "test.txt"
            input_file.write_text("Meeting notes here")

            runner = CliRunner()
            result = runner.invoke(main, ['process', str(input_file), '--output', str(tmp_path), '--no-dedup'])

            assert result.exit_code == 0

    def test_process_with_verbose_flag(self, tmp_path):
        """Process with --verbose flag."""
        with patch('minutes.cli_process.get_backend') as mock_get_backend:
            mock_backend = Mock()
            extraction_result = ExtractionResult(
                decisions=[],
                ideas=[],
                questions=[],
                action_items=[],
                concepts=[],
                terms=[],
                tldr="Test TLDR"
            )
            mock_backend.generate.return_value = json.dumps(extraction_result.model_dump())
            mock_get_backend.return_value = (mock_backend, "mock")

            input_file = tmp_path / "test.txt"
            input_file.write_text("Meeting notes here")

            runner = CliRunner()
            result = runner.invoke(main, ['process', str(input_file), '--output', str(tmp_path), '--verbose'])

            assert result.exit_code == 0


class TestConfigCommand:
    """Test the config command."""

    def test_config_prints_settings(self):
        """Config command prints configuration values."""
        runner = CliRunner()
        result = runner.invoke(main, ['config'])
        assert result.exit_code == 0
        # Should print some config info
        assert len(result.output) > 0

    def test_config_with_env_flag(self):
        """Config command with --env flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['config', '--env'])
        assert result.exit_code == 0


class TestWatchCommand:
    """Test the watch command."""

    def test_watch_validates_directory(self, tmp_path):
        """Watch command validates directory exists."""
        runner = CliRunner()
        # Test with nonexistent directory
        result = runner.invoke(main, ['watch', '/nonexistent/path'])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


class TestCLIHelp:
    """Test CLI help and structure."""

    def test_main_help(self):
        """Main CLI group shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'process' in result.output
        assert 'watch' in result.output
        assert 'config' in result.output

    def test_process_help(self):
        """Process command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['process', '--help'])
        assert result.exit_code == 0
        assert 'output' in result.output or '--output' in result.output

    def test_watch_help(self):
        """Watch command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['watch', '--help'])
        assert result.exit_code == 0

    def test_config_help(self):
        """Config command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['config', '--help'])
        assert result.exit_code == 0
