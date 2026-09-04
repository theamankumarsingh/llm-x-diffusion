# Copyright 2026 Aman Kumar Singh
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

import json
from pathlib import Path

def save_results(img, info, timestamp, root="artifacts"):
    run = Path(root) / timestamp
    run.mkdir(parents=True, exist_ok=True)
    img.save(run / "output.png")
    (run / "info.json").write_text(json.dumps(info, indent=2))
