import os
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from uuid import UUID
from warnings import warn

import anyio
import requests
import uvicorn
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from rungrid.cli import CLI
from rungrid.experiment import StaleTrial, Trial, TrialResult, VarNamespace
from rungrid.plumbing import PredicateType, Sink, Source
from tenacity import retry, stop_after_attempt, wait_random_exponential

from analise_db_iasmin_2026.options import set_options

load_dotenv()


def get_secret_key() -> bytes:
    dev_secret = b"nzin34T4nScsSXOVvgEk47nwi-jHWBT6zt1rfW5xAZg="
    try:
        return os.environb[b"SECRET_KEY"]
    except KeyError:
        warn(
            "WARNING: using dev secret. please set the SECRET_KEY environment "
            + "variable with the output of cryptography.Fernet.generate_key()",
            UserWarning,
        )
        return dev_secret


@dataclass
class MsgAskTrial:
    options: dict[str, str]


@dataclass
class MsgTrial:
    trial: StaleTrial


@dataclass
class MsgTellTrial:
    trial_number: int | None
    uuid: UUID
    result: TrialResult


class TrialClient(Source, Sink):
    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._fernet = Fernet(get_secret_key())

    def _fetch_trial(self) -> StaleTrial:
        opts = CLI.get_instance().get_options()
        msg = MsgAskTrial(options=opts)
        send_payload = self._fernet.encrypt(pickle.dumps(msg))

        @retry(stop=stop_after_attempt(5), wait=wait_random_exponential())
        def post():
            return requests.post(self._url, data=send_payload)

        resp = post()
        recv_payload: MsgTrial = pickle.loads(self._fernet.decrypt(resp.content))
        assert isinstance(recv_payload, MsgTrial)
        return recv_payload.trial

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        if finished:
            return
        if search_tag:
            return
        while 1:
            yield self._fetch_trial()

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        return None

    def put_trial(self, trial: Trial) -> None:
        if trial.result is None:
            return
        msg = MsgTellTrial(
            trial_number=trial.optuna_trial_number, result=trial.result, uuid=trial.uuid
        )
        send_payload = self._fernet.encrypt(pickle.dumps(msg))

        @retry(stop=stop_after_attempt(5), wait=wait_random_exponential())
        def post():
            requests.post(self._url, data=send_payload)

        post()


class TrialServer:
    def __init__(self, source: Source, sink: Sink) -> None:
        super().__init__()
        self._fastapi = FastAPI()
        self._fastapi.post("/")(self._on_get)
        self._fernet = Fernet(get_secret_key())
        self._source = source
        self._sink = sink
        self._source_iter: Iterator[Trial] | None = None

    async def _get_trial(self, opts: dict[str, str]) -> Trial:
        while 1:
            if self._source_iter is None:
                self._source_iter = iter(self._source.get_trials())
            try:
                with set_options(opts):
                    return next(self._source_iter)
            except StopIteration:
                self._source_iter = None
            await anyio.sleep(1)

    async def _on_get(self, req: Request) -> Response:
        body = await req.body()
        msg = pickle.loads(self._fernet.decrypt(body))
        if isinstance(msg, MsgAskTrial):
            trial = await self._get_trial(msg.options)
            resp = MsgTrial(trial=trial.as_stale())
            return Response(content=self._fernet.encrypt(pickle.dumps(resp)))
        elif isinstance(msg, MsgTellTrial):
            trial = StaleTrial(
                msg.trial_number, msg.uuid, VarNamespace(set())
            ).with_result(msg.result)
            self._sink.put_trial(trial)
            return Response()
        return Response(status_code=400)

    def serve(self) -> None:
        uvicorn.run(self._fastapi)
