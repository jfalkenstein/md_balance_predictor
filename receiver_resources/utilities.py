from datetime import date, timedelta
from math import ceil

from receiver_resources.enums import ValueConsistency


def get_week_of_month(date_to_evaluate: date) -> int:
    first_day_of_month = date_to_evaluate.replace(day=1)
    day_of_month = date_to_evaluate.day
    adjusted_day_of_month = (day_of_month + first_day_of_month.weekday() + 1)
    week_of_month = int(ceil(adjusted_day_of_month / 7))
    return week_of_month


def map_value_consistency(cv_value):
    if cv_value < .1:
        return ValueConsistency.EXTREMELY_CONSISTENT
    if cv_value < .3:
        return ValueConsistency.RELATIVELY_CONSISTENT
    if cv_value < .5:
        return ValueConsistency.SOME_VARIANCE
    return ValueConsistency.VERY_VARIANT


def iterate_days(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)