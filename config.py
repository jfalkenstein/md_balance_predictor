from datetime import datetime
from decimal import Decimal

FILE_NAME = 'data.json'
REQUIRED_TXN_FREQUENCY = .3
REQUIRED_TXN_PERCENT_OF_MONTHS = .3
MAX_FREQUENCY_FOR_MONTHLY_RECURRANCE = 5


def get_config(moneybot):
    class Config:
        DEFAULT_ACCOUNT = 'US Bank Checking'
        SELECTED_ACCOUNT_NAME = moneybot.getField(
            None,
            None,
            'Account to Predict',
            'What account name should be predicted? Defaults to "{}".'.format(DEFAULT_ACCOUNT),
            None,
            False
        ) or DEFAULT_ACCOUNT

        DEFAULT_MONTH_COUNT = 9
        SELECTED_MONTH_COUNT = int(
            moneybot.getField(
                None,
                None,
                'Number of Months to Analyze',
                'How many complete months of history should be analyzed? Defaults to {}.'.format(DEFAULT_MONTH_COUNT),
                None,
                False
            ) or DEFAULT_MONTH_COUNT
        )

        ANALYSIS_PERIOD_DAYS = 30 * SELECTED_MONTH_COUNT
        START_DATE = moneybot.getField(
            None,
            None,
            'Start Date (YYYY-MM-DD)',
            'What date should the predictor start on? If nothing specified, defaults to today.',
            None,
            False
        )
        START_DATE = datetime.strptime(START_DATE, '%Y-%m-%d').date() if START_DATE else datetime.now().date()

        DEFAULT_DAYS_TO_PREDICT = 30
        DAYS_TO_PREDICT = int(
            moneybot.getField(
                None,
                None,
                'Days to predict',
                'How many days out to predict? Defaults to {}'.format(DEFAULT_DAYS_TO_PREDICT),
                None,
                False
            ) or DEFAULT_DAYS_TO_PREDICT
        )

        START_DATE_BALANCE = Decimal(
            moneybot.getField(
                None,
                None,
                'Current Bank Balance',
                'What is the bank account balance as of {}? Defaults to 0.'.format(START_DATE),
                None,
                False
            ) or 0
        )

    return Config
