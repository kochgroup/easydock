"""
Regression tests for the streaming Uni-pKa pipeline (containers/unipka/unipka.py).

They check the bookkeeping of UnipkaStream only, therefore the free energy predictor is
replaced by a stub and neither the model nor a GPU is required.
"""
import sys
from pathlib import Path

import pytest

UNIPKA_DIR = Path(__file__).resolve().parents[1] / 'containers' / 'unipka'
TEMPLATE_FILE = str(UNIPKA_DIR / 'simple_smarts_pattern.tsv')

sys.path.insert(0, str(UNIPKA_DIR))
unipka = pytest.importorskip('unipka', reason='dependencies of unipka.py are not installed')


class StubPredictor:
    """Returns a fixed free energy for every microstate and records all requests."""

    def __init__(self):
        self.requests = []

    def predict(self, smiles_list):
        smiles_list = list(smiles_list)
        self.requests.append(smiles_list)
        return {smi: 0.0 for smi in smiles_list}

    @property
    def requested_microstates(self):
        return [smi for request in self.requests for smi in request]


@pytest.fixture(scope='module')
def templates():
    return unipka.read_template(TEMPLATE_FILE)


@pytest.fixture(scope='module')
def patterns():
    return unipka.get_template_patterns(TEMPLATE_FILE)


def run_stream(items, templates, patterns, **kwargs):
    predictor = StubPredictor()
    stream = unipka.UnipkaStream(template_a2b=templates[0], template_b2a=templates[1],
                                 predictor=predictor, patterns=patterns, ncpu=1, **kwargs)
    return list(stream.process(iter(items))), predictor


# input molecules which are the same compound in different protonation states, their
# ensembles consist of exactly the same microstates
PROTONATION_PAIRS = [('CC(=O)O', 'acid_neutral'), ('CC(=O)[O-]', 'acid_anion'),
                     ('CCN', 'amine_neutral'), ('CC[NH3+]', 'amine_cation')]

# input molecules written as non-canonical SMILES of the same compound
SPELLING_DUPLICATES = [('OCC', 'ethanol_1'), ('CCO', 'ethanol_2'),
                       ('c1ccccc1C(=O)O', 'benzoic_1'), ('OC(=O)c1ccccc1', 'benzoic_2')]

DISTINCT_MOLECULES = [('c1ccccc1O', 'phenol'), ('CC(N)=O', 'acetamide'),
                      ('NCCc1ccccc1', 'phenethylamine'), ('OC(=O)CCC(=O)O', 'succinic')]


@pytest.mark.parametrize('items', [PROTONATION_PAIRS, SPELLING_DUPLICATES, DISTINCT_MOLECULES,
                                   PROTONATION_PAIRS + DISTINCT_MOLECULES])
def test_every_input_molecule_is_returned_once(items, templates, patterns):
    """Molecules sharing microstates must not be dropped or duplicated."""
    results, _ = run_stream(items, templates, patterns)
    assert sorted(res.name for res in results) == sorted(name for _, name in items)


@pytest.mark.parametrize('items', [PROTONATION_PAIRS, SPELLING_DUPLICATES])
def test_shared_microstates_are_predicted_once(items, templates, patterns):
    """A microstate shared by several parents is sent to the predictor only once."""
    _, predictor = run_stream(items, templates, patterns)
    requested = predictor.requested_microstates
    assert len(requested) == len(set(requested))


@pytest.mark.parametrize('items', [PROTONATION_PAIRS, SPELLING_DUPLICATES, DISTINCT_MOLECULES])
def test_shared_microstates_do_not_affect_forms(items, templates, patterns):
    """Molecules sharing microstates get the same forms as if processed on their own."""
    results, _ = run_stream(items, templates, patterns)
    forms = {res.name: res.forms for res in results}
    for item in items:
        alone, _ = run_stream([item], templates, patterns)
        assert len(alone) == 1
        assert forms[item[1]] == alone[0].forms


def test_identical_smiles_are_returned_once_each(templates, patterns):
    """
    The same SMILES supplied twice must produce exactly one result per input line, also when
    the second copy arrives after the first one has already been completed and reported (a
    small priority buffer and an immediate GPU trigger reproduce this ordering).
    """
    items = [('CC(=O)O', 'first'), ('c1ccccc1O', 'phenol'), ('CCN', 'amine'),
             ('CC(=O)O', 'second')]
    results, _ = run_stream(items, templates, patterns, priority_buffer_size=1,
                            gpu_trigger_microstates=1, gpu_trigger_timeout=0)
    assert sorted(res.name for res in results) == ['amine', 'first', 'phenol', 'second']
    forms = {res.name: res.forms for res in results}
    assert forms['first'] == forms['second']


def test_molecules_without_microstates_are_returned(templates, patterns):
    """A molecule with an empty ensemble is reported with no forms instead of being dropped."""
    items = [('C', 'methane'), ('c1ccccc1O', 'phenol')]
    results, _ = run_stream(items, templates, patterns)
    assert sorted(res.name for res in results) == ['methane', 'phenol']


def test_molecules_without_predicted_energies_are_returned(templates, patterns):
    """A molecule whose microstates all fail the prediction is returned with no forms."""
    class FailingPredictor(StubPredictor):
        def predict(self, smiles_list):
            super().predict(smiles_list)
            return {}   # no microstate gets a free energy

    predictor = FailingPredictor()
    stream = unipka.UnipkaStream(template_a2b=templates[0], template_b2a=templates[1],
                                 predictor=predictor, patterns=patterns, ncpu=1)
    results = list(stream.process(iter(DISTINCT_MOLECULES)))
    assert sorted(res.name for res in results) == sorted(name for _, name in DISTINCT_MOLECULES)
    assert all(res.forms == [] for res in results)


def test_stalled_molecule_is_reported_with_no_forms(templates, patterns, caplog):
    """
    A molecule which never collects all its microstates must not be silently abandoned: it
    has to be returned with no forms and reported as an error.
    """
    class LosingStream(unipka.UnipkaStream):
        """Loses one microstate per flush, so its molecule can never be completed."""
        def _flush_gpu(self, microstate_queue, pending, smi_to_names):
            return super()._flush_gpu(microstate_queue[1:], pending, smi_to_names)

    stream = LosingStream(template_a2b=templates[0], template_b2a=templates[1],
                          predictor=StubPredictor(), patterns=patterns, ncpu=1)
    with caplog.at_level('ERROR', logger='uni-pka'):
        results = list(stream.process(iter(DISTINCT_MOLECULES)))
    assert sorted(res.name for res in results) == sorted(name for _, name in DISTINCT_MOLECULES)
    stalled = [res for res in results if not res.forms]
    assert len(stalled) == 1
    assert 'was left unfinished' in caplog.text
