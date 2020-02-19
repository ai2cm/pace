import logging


logger = logging.getLogger('fv3util-tests')


class DummyComm:

    def __init__(self, rank, total_ranks, buffer_dict):
        self.rank = rank
        self.total_ranks = total_ranks
        self._buffer = buffer_dict
        self._i_buffer = {}

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.total_ranks

    def _get_buffer(self, buffer_type, in_value):
        i_buffer = self._i_buffer.get(buffer_type, 0)
        self._i_buffer[buffer_type] = i_buffer + 1
        if buffer_type not in self._buffer:
            self._buffer[buffer_type] = []
        if self.rank == 0:
            self._buffer[buffer_type].append(in_value)
        return self._buffer[buffer_type][i_buffer]

    @property
    def _bcast_buffer(self):
        if 'bcast' not in self._buffer:
            self._buffer['bcast'] = []
        return self._buffer['bcast']

    @property
    def _scatter_buffer(self):
        if 'scatter' not in self._buffer:
            self._buffer['scatter'] = []
        return self._buffer['scatter']

    def bcast(self, value, root=0):
        if root != 0:
            raise NotImplementedError('DummyComm assumes ranks are called in order, so root must be the scatter source')
        value = self._get_buffer('bcast', value)
        logger.debug(f'bcast {value} to rank {self.rank}')
        return value

    def barrier(self):
        return

    def Scatter(self, sendbuf, recvbuf, root=0):
        if root != 0:
            raise NotImplementedError('DummyComm assumes ranks are called in order, so root must be the scatter source')
        sendbuf = self._get_buffer('scatter', sendbuf)
        recvbuf[:] = sendbuf[self.rank]
