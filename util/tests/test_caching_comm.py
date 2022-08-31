import copy
import io
from typing import List

import numpy as np

import pace.util
from pace.util.caching_comm import CachingCommReader, CachingCommWriter


def test_halo_update_integration():
    shape = (18, 18)
    dims = [pace.util.X_DIM, pace.util.Y_DIM]
    origin = (3, 3)
    extent = (12, 12)
    n_ranks = 6
    partitioner = pace.util.CubedSpherePartitioner(
        tile=pace.util.TilePartitioner(layout=(1, 1))
    )
    quantity_list = [
        pace.util.Quantity(
            data=np.random.randn(*shape),
            dims=dims,
            units="",
            origin=origin,
            extent=extent,
        )
        for _ in range(n_ranks)
    ]
    buffer_dict = {}
    write_communicator_list: List[pace.util.CubedSphereCommunicator] = []
    for i in range(n_ranks):
        write_communicator_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.CachingCommWriter(
                    pace.util.LocalComm(
                        rank=i, total_ranks=n_ranks, buffer_dict=buffer_dict
                    )
                ),
                partitioner=partitioner,
            )
        )
    local_comm_quantities = copy.deepcopy(quantity_list)
    perform_serial_halo_updates(write_communicator_list, local_comm_quantities)

    read_communicator_list: List[pace.util.CubedSphereCommunicator] = []
    for i in range(n_ranks):
        file = io.BytesIO()
        write_communicator_list[i].comm.dump(file)
        file.seek(0)
        read_communicator_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.CachingCommReader.load(file),
                partitioner=partitioner,
            )
        )
    perform_serial_halo_updates(read_communicator_list, quantity_list)
    for local_comm_quantity, read_quantity in zip(local_comm_quantities, quantity_list):
        np.testing.assert_array_equal(local_comm_quantity.data, read_quantity.data)


def perform_serial_halo_updates(
    communicator_list: List[pace.util.CubedSphereCommunicator],
    quantity_list: List[pace.util.Quantity],
):
    req_list = []
    for communicator, quantity in zip(communicator_list, quantity_list):
        req_list.append(communicator.start_halo_update(quantity, n_points=3))
    for req in req_list:
        req.wait()


def test_Recv_inserts_data():
    comm = pace.util.CachingCommWriter(
        comm=pace.util.NullComm(rank=0, total_ranks=6, fill_value=0.0)
    )
    shape = (12, 12)
    recvbuf = np.random.randn(*shape)
    assert len(comm._data.received_buffers) == 0
    comm.Recv(recvbuf, source=0)
    assert len(comm._data.received_buffers) == 1
    assert comm._data.received_buffers[0].shape == shape


def test_Irecv_inserts_data():
    comm = pace.util.CachingCommWriter(
        comm=pace.util.NullComm(rank=0, total_ranks=6, fill_value=0.0)
    )
    shape = (12, 12)
    recvbuf = np.random.randn(*shape)
    assert len(comm._data.received_buffers) == 0
    req = comm.Irecv(recvbuf, source=0)
    assert len(comm._data.received_buffers) == 0
    req.wait()
    assert len(comm._data.received_buffers) == 1
    assert comm._data.received_buffers[0].shape == shape


def test_bcast_inserts_data():
    comm = pace.util.CachingCommWriter(
        comm=pace.util.NullComm(rank=0, total_ranks=6, fill_value=0.0)
    )
    shape = (12, 12)
    recvbuf = np.random.randn(*shape)
    assert len(comm._data.bcast_objects) == 0
    comm.bcast(recvbuf)
    assert len(comm._data.bcast_objects) == 1
    assert comm._data.bcast_objects[0].shape == shape
    np.testing.assert_array_equal(comm._data.bcast_objects[0], recvbuf)
    assert comm._data.bcast_objects[0] is not recvbuf


def writer_to_reader(comm_writer: CachingCommWriter) -> CachingCommReader:
    file = io.BytesIO()
    comm_writer.dump(file)
    file.seek(0)
    return CachingCommReader.load(file)


def test_Scatter():
    np.random.seed(0)
    array = np.random.uniform(size=(50,))
    send_array = np.empty([2] + list(array.shape))
    send_array[:] = array[None, :]
    buffer_dict = {}
    root_comm = pace.util.CachingCommWriter(
        comm=pace.util.LocalComm(rank=0, total_ranks=2, buffer_dict=buffer_dict)
    )
    worker_comm = pace.util.CachingCommWriter(
        comm=pace.util.LocalComm(rank=1, total_ranks=2, buffer_dict=buffer_dict)
    )
    recvbuf_root = np.zeros_like(array)
    recvbuf_worker = np.zeros_like(array)
    root_comm.Scatter(send_array, recvbuf=recvbuf_root, root=0)
    worker_comm.Scatter(None, recvbuf=recvbuf_worker, root=0)
    np.testing.assert_array_equal(recvbuf_root, recvbuf_worker)

    root_comm = writer_to_reader(root_comm)
    worker_comm = writer_to_reader(worker_comm)

    recvbuf2_root = np.zeros_like(array)
    recvbuf2_worker = np.zeros_like(array)
    root_comm.Scatter(send_array, recvbuf=recvbuf2_root, root=0)
    worker_comm.Scatter(None, recvbuf=recvbuf2_worker, root=0)
    np.testing.assert_array_equal(recvbuf2_root, recvbuf_root)
    np.testing.assert_array_equal(recvbuf2_worker, recvbuf_worker)


def test_Gather():
    np.random.seed(0)
    array_root = np.random.uniform(size=(50,))
    array_worker = np.random.uniform(size=(50,))
    buffer_dict = {}
    root_comm = pace.util.CachingCommWriter(
        comm=pace.util.LocalComm(rank=0, total_ranks=2, buffer_dict=buffer_dict)
    )
    worker_comm = pace.util.CachingCommWriter(
        comm=pace.util.LocalComm(rank=1, total_ranks=2, buffer_dict=buffer_dict)
    )
    recvbuf_root = np.empty([2] + list(array_root.shape))
    worker_comm.Gather(array_worker, recvbuf=None, root=0)
    root_comm.Gather(array_root, recvbuf=recvbuf_root, root=0)
    np.testing.assert_array_equal(recvbuf_root[0, :], array_root)
    np.testing.assert_array_equal(recvbuf_root[1, :], array_worker)

    root_comm = writer_to_reader(root_comm)
    worker_comm = writer_to_reader(worker_comm)

    recvbuf_root = np.empty([2] + list(array_root.shape))
    worker_comm.Gather(array_worker, recvbuf=None, root=0)
    root_comm.Gather(array_root, recvbuf=recvbuf_root, root=0)
    np.testing.assert_array_equal(recvbuf_root[0, :], array_root)
    np.testing.assert_array_equal(recvbuf_root[1, :], array_worker)
