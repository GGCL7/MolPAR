"""公共工具: 配置加载、路径解析、限速且域名锁定的 KEGG HTTP 客户端。

安全说明 (SSRF 防护): 只允许访问 config 中 kegg.base_url 指定的官方公网域名
(rest.kegg.jp)。任何其它主机/内网地址一律拒绝。
"""
from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml
import requests

# data_build/ 根目录 = 本文件(scripts/)的上一级
ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def rp(cfg: dict, key: str) -> Path:
    """解析 paths.<key> 为绝对路径并确保目录存在。"""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


class KeggClient:
    """限速 + 重试 + 域名白名单的 KEGG REST 客户端。"""

    def __init__(self, cfg: dict):
        k = cfg["kegg"]
        self.base = k["base_url"].rstrip("/")
        self.allowed_host = urlparse(self.base).hostname
        self.min_interval = 1.0 / float(k["rate_limit_per_sec"])
        self.timeout = k["timeout"]
        self.max_retries = k["max_retries"]
        self.backoff = k["retry_backoff"]
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MVP-MP-dataset-builder/1.0"})

    def _guard(self, url: str):
        host = urlparse(url).hostname
        if host != self.allowed_host:
            raise ValueError(f"拒绝访问非白名单主机: {host} (仅允许 {self.allowed_host})")

    def _throttle(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()

    def get(self, path: str) -> str | None:
        """GET base_url + path, 返回文本; 404/空 返回 None。"""
        url = f"{self.base}/{path.lstrip('/')}"
        self._guard(url)
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    return None
                if r.status_code == 200:
                    return r.text
                # 400/403/500 等: 重试
                sys.stderr.write(f"[warn] {r.status_code} {url} (try {attempt})\n")
            except requests.RequestException as e:
                sys.stderr.write(f"[warn] {e} {url} (try {attempt})\n")
            time.sleep(self.backoff ** attempt)
        sys.stderr.write(f"[error] 放弃: {url}\n")
        return None


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
