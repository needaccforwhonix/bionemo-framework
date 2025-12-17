# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import pytest


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture
def tokenizer_path(recipe_path):
    """Get the path to the nucleotide tokenizer."""
    return str(recipe_path / "tokenizers" / "nucleotide_fast_tokenizer")


@pytest.fixture(autouse=True)
def debug_api_cleanup():
    """Ensure nvdlfw_inspect does not stay initialized across tests."""
    yield
    try:
        import nvdlfw_inspect.api as debug_api

        debug_api.end_debug()
    except Exception:  # pragma: no cover - best-effort cleanup for optional dependency
        pass


def pytest_collection_modifyitems(items):
    """Run FP8 stats logging tests first to avoid late debug initialization."""
    stats_test_names = {
        "test_sanity_ddp_fp8_stats_logging",
        "test_sanity_fsdp2_fp8_stats_logging",
    }
    stats_tests = [item for item in items if item.name in stats_test_names]
    other_tests = [item for item in items if item.name not in stats_test_names]
    items[:] = stats_tests + other_tests
