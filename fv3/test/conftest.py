def pytest_addoption(parser):
    parser.addoption("--which_modules", action="store", default="all")
    parser.addoption("--skip_modules", action="store", default="none")
    parser.addoption("--print_failures", action="store_true")
    parser.addoption("--failure_stride", action="store", default=1)
    parser.addoption("--data_path", action="store", default="./")
    parser.addoption("--data_backend", action="store", default="numpy")
    parser.addoption("--exec_backend", action="store", default="numpy")
    parser.addoption("--backend", action="store", default="numpy")
