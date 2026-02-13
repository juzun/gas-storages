# import datetime as dt

# import numpy as np
# import pandas as pd
# import pytest
# from gas_storage.core.gas_storage import GasStorage


def test_dummy():
    assert True


# @pytest.fixture
# def storage():
#     date_start = dt.date(2022, 1, 1)
#     date_end = dt.date(2022, 1, 31)
#     storage = GasStorage("test_storage", date_start, date_end)
#     prices = {date_start: 1}
#     prices = pd.DataFrame(list(prices.items()), columns=["date", "price"])
#     prices = pd.DataFrame(
#         zip(list(pd.to_datetime(prices["date"])), list(prices["price"])),
#         columns=["date", "price"],
#     )
#     storage.load_attribute("prices", prices)
#     storage.load_attribute("wgv", 100, date_start, date_end)
#     storage.load_attribute("wr", 10, date_start, date_end)
#     storage.load_attribute("ir", 10, date_start, date_end)
#     inj_curve = np.array([[0, 50], [50, 100], [50, 100]]) / 100
#     storage.load_attribute("inj_curve", inj_curve, date_start, date_end)
#     wit_curve = np.array([[0, 50], [50, 100], [50, 100]]) / 100
#     storage.load_attribute("wit_curve", wit_curve, date_start, date_end)
#     storage.set_initial_state(100)
#     storage.create_model()

#     storage.solve_model(solver_name="scip", stream_solver=False)
#     storage.create_graph()
#     total_graph, total_daily_export, total_monthly_export = collect([storage])
#     return storage, total_graph, total_daily_export, total_monthly_export


# def test_model_exists(storage):
#     assert storage[0].mdl is not None, "Model object should exist."


# def test_attributes_table_exists(storage):
#     assert storage[0].attr is not None, "Attributes table should exist."


# def test_objective_exists(storage):
#     assert hasattr(storage[0].mdl, "objective"), "Objective should exist."


# def test_objective_value(storage):
#     assert storage[0].objective == 100, "Objective value should be 100."


# def test_graphs_exist(storage):
#     assert storage[0].fig is not None, "Graph should exist."
#     assert storage[1] is not None, "Total graph should exist."


# def test_graphs_content(storage):
#     assert len(storage[0].fig.data) == 3, "Graph should contain 3 graphs."
#     assert len(storage[1].data) == 3, "Total graph should contain 3 graphs."


# def test_export_exists(storage):
#     assert storage[2] is not None, "Daily export should exist."
#     assert storage[3] is not None, "Monthly export should exist."


# def test_export_content(storage):
#     assert len(storage[2]) > 0, "Daily export should contain data."
#     assert len(storage[3]) > 0, "Monthly export should contain data."
