from fv3util import get_tile_number
import unittest


class TileNumberTests(unittest.TestCase):
    
    def test_get_tile_number_six_ranks(self):
        rank_list = list(range(6))
        for i_rank in rank_list:
            with self.subTest(msg=i_rank):
                tile = get_tile_number(i_rank, len(rank_list))
                self.assertEqual(tile, i_rank + 1)

    def test_get_tile_number_twenty_four_ranks(self):
        rank_list = list(range(24))
        i_rank = 0
        for i_tile in [i + 1 for i in range(6)]:
            for _ in range(4):
                return_value = get_tile_number(i_rank, len(rank_list))
                self.assertEqual(return_value, i_tile)
                i_rank += 1


if __name__ == '__main__':
    unittest.main()
