import pytest
from milton.icd10 import ICD10Tree


@pytest.fixture(scope='module')
def ICD10():
    return ICD10Tree.load()


@pytest.fixture
def test_nodes(ICD10):
    return [
        ICD10,
        ICD10.Chapter_XIV,
        ICD10.Chapter_XIV.N17_N19,
        ICD10.Chapter_XIV.N17_N19.N18,
        ICD10.Chapter_XIV.N17_N19.N18.N185
    ]


def test_code_listing(ICD10):
    node = ICD10.Chapter_XIV.N17_N19.N18
    expected_codes = ['N180', 'N181', 'N182', 'N183',
                      'N184', 'N185', 'N188', 'N189']
    assert list(node) == expected_codes
    assert len(node.subsections()) == len(list(node))

    for subnode, expected_code in zip(node.subsections(), expected_codes):
        assert list(subnode) == [expected_code]


def test_subsections(ICD10, test_nodes):
    for node in test_nodes:
        l2_subsections = set()

        # collect level2 subsections "manually"
        for subnode in node.subsections():
            l2_subsections.update(subnode.subsections())

        assert set(node.subsections(level=2)) == l2_subsections


def test_node_names(ICD10):
    assert ICD10.name == 'ICD10'
    assert ICD10.Chapter_XIV.name == 'Chapter XIV'
    assert ICD10.Chapter_XIV.N17_N19.name == 'N17-N19'
    assert ICD10.Chapter_XIV.N17_N19.N18.name == 'N18'
    assert ICD10.Chapter_XIV.N17_N19.N18.N185.name == 'N185'


def test_name_descr_mapping(ICD10, test_nodes):
    for node in test_nodes:
        assert ICD10[node.name] == str(node)


def test_chapter_property(ICD10, test_nodes):
    for node in test_nodes:
        if node is ICD10:
            assert node.chapter is None
        else:
            assert node.chapter is ICD10.Chapter_XIV


@pytest.mark.parametrize('code, full_code, descr', [
    ('D471', 'D47.1', 'Chronic myeloproliferative disease'),
    ('N180', 'N18.0', 'End-stage renal disease'),
    ('Z853', 'Z85.3', 'Personal history of malignant neoplasm of breast')
])
def test_codes_with_dots(ICD10, code, full_code, descr):
    node = ICD10.find_by_code(code)
    assert node.name == code
    assert node.full_name == full_code
    assert node.description == descr
