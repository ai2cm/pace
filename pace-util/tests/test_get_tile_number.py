import pytest

from pace.util import get_tile_number


@pytest.mark.cpu_only
def test_get_tile_number_six_ranks():
    rank_list = list(range(6))
    for i_rank in rank_list:
        tile = get_tile_number(i_rank, len(rank_list))
        assert tile == i_rank + 1


@pytest.mark.cpu_only
def test_get_tile_number_twenty_four_ranks():
    rank_list = list(range(24))
    i_rank = 0
    for i_tile in [i + 1 for i in range(6)]:
        for _ in range(4):
            return_value = get_tile_number(i_rank, len(rank_list))
            assert return_value == i_tile
            i_rank += 1
