import logging


logger = logging.getLogger('fv3util-tests')


class DummyComm:

    def __init__(self, rank, total_ranks, buffer_dict):
        self.rank = rank
        self.total_ranks = total_ranks
        self._buffer = buffer_dict
        self._i_buffer = {}
        self._split_comms = {}
        self._split_buffers = {}

    def __repr__(self):
        return f"DummyComm(rank={self.rank}, total_ranks={self.total_ranks})"

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

    def _get_send_recv(self, from_rank):
        self._buffer['send_recv'] = self._buffer.get('send_recv', {})
        return self._buffer['send_recv'].pop((from_rank, self.rank))

    def _put_send_recv(self, value, to_rank):
        self._buffer['send_recv'] = self._buffer.get('send_recv', {})
        self._buffer['send_recv'][(self.rank, to_rank)] = value

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
            raise NotImplementedError('DummyComm assumes ranks are called in order, so root must be the bcast source')
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

    def Send(self, sendbuf, dest):
        self._put_send_recv(sendbuf, dest)

    def Isend(self, sendbuf, dest):
        return self.Send(sendbuf, dest)

    def Recv(self, recvbuf, source):
        recvbuf[:] = self._get_send_recv(source)

    def Irecv(self, recvbuf, source):
        return self.Recv(recvbuf, source)

    def Split(self, color, key):
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])
        self._split_buffers[color] = self._split_buffers.get(color, {})
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm = DummyComm(
            rank=rank,
            total_ranks=total_ranks,
            buffer_dict=self._split_buffers[color]
        )
        for comm in self._split_comms[color]:
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm
