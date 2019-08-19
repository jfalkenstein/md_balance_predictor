#!/usr/bin/env python
# Python script to be run in Moneydance to perform amazing feats of financial scripting
from __future__ import unicode_literals, print_function, division, absolute_import

import json
import os
import sys
from decimal import Decimal
from subprocess import check_call, Popen
from datetime import timedelta, datetime

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

from com.infinitekind.moneydance.model import (
    AccountUtil,
    AcctFilter,
    TxnSearch,
    Account
)
from config import FILE_NAME, get_config

if sys.version_info[0] == 3:
    unicode = str


def run_script():
    # get the default environment variables, set by Moneydance
    # print "The Moneydance app controller: %s" % (moneydance)
    print("The current data set: %s" % (moneydance_data))
    # print "The UI: %s" % (moneydance_ui)
    # print "Bot interface: %s" % (moneybot)
    config = get_config(moneybot)
    main_account = get_bank_accounts(moneydance_data, config)
    checking_account_transactions = list(
        iterate_transactions_within_specified_period(main_account, moneydance_data, config)
    )
    categories_for_transactions = get_categories_for_transactions(checking_account_transactions)

    cash_flow_transactions = get_cashflow_transactions(checking_account_transactions)

    data_to_send = {
        'accounts': [c.to_dict() for c in categories_for_transactions],
        'transactions': [t.to_dict() for t in cash_flow_transactions],
        'current_account_balance': unicode(config.START_DATE_BALANCE),
        'start_date': config.START_DATE.isoformat(),
        'days_to_predict': config.DAYS_TO_PREDICT
    }

    jsonified = json.dumps(data_to_send, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    send_to_cpython(jsonified)


def get_bank_accounts(account_book, config):
    root_account = account_book.getRootAccount()
    account_filter = AccountFilter(config.SELECTED_ACCOUNT_NAME)
    return root_account.getSubAccounts(account_filter)


def iterate_transactions_within_specified_period(accounts, account_book, config):
    master_transaction_set = account_book.getTransactionSet()
    for account in accounts:
        transactions_for_main_account = master_transaction_set.getTransactionsForAccount(account)
        transactions_for_main_account.sortByField(AccountUtil.DATE)
        filterer = FullMonthDateRangeFilter(
            config.ANALYSIS_PERIOD_DAYS,
            config.START_DATE
        )
        for txn in transactions_for_main_account.iterator():
            if not filterer.matches(txn):
                continue
            transaction = Transaction(txn)
            if transaction.is_pre_payment_transfer or transaction.is_inter_account_transfer:
                continue
            yield transaction


def get_categories_for_transactions(transaction_iterator):
    categories = set()
    for transaction in transaction_iterator:
        for other_transaction in transaction.other_transactions:  # type: Transaction
            if transaction.account in other_transaction.account.account_path:
                continue
            categories.add(other_transaction.account)
    return categories


def get_cashflow_transactions(transaction_iterator):
    transactions = []
    for transaction in transaction_iterator:
        for other_transaction in transaction.other_transactions:
            if transaction.account in other_transaction.account.account_path:
                continue
            if other_transaction.account.account_type == Account.AccountType.BANK:
                continue
            transactions.append(other_transaction)
    return transactions


def send_to_cpython(pickled_data):
    print("Sending data to cPython...")
    scripts_path = os.path.join(current_directory, 'Scripts')
    python_path = os.path.join(current_directory, 'Scripts', 'python')
    if not os.path.exists(scripts_path):
        print("Creating virtual environment...")
        requirements_path = os.path.join(current_directory, 'requirements.txt')
        check_call(['python', '-m', 'venv', current_directory])
        check_call([python_path, '-m', 'pip', 'install', '-r', requirements_path])

    receiver_file = os.path.join(current_directory, 'receiver.py')
    filepath = os.path.join(current_directory, FILE_NAME)
    command = [python_path, receiver_file, 'handshake']
    with open(filepath, mode='w') as f:
        f.write(pickled_data)
        f.flush()
    process = Popen(command)

    process.wait()

def convert_date_int_to_date_object(date_int):
    date_str = str(date_int)
    date = datetime.strptime(date_str, '%Y%m%d').date()
    return date


def convert_value_to_decimal(long_value):
    str_value = str(long_value)
    str_value = str_value[:-1] if str_value.endswith('L') else str_value
    formatted = '{}.{}'.format(str_value[:-2], str_value[-2:])
    decimal = Decimal(formatted)
    return decimal

class AccountFilter(AcctFilter):
    def __init__(self, account_name):
        self.account_name = account_name

    def format(self, account):
        return account.getAccountName()

    def matches(self, account):
        return self.format(account) == self.account_name


class FullMonthDateRangeFilter(TxnSearch, object):
    def __init__(self, relative_date_range, start_date):
        a_month_ago = start_date - timedelta(days=30)
        time_range = timedelta(days=relative_date_range)
        threshhold_day = a_month_ago - time_range
        self.full_month_range_start_day = threshhold_day.replace(day=1)
        self.full_month_range_end_day = start_date.replace(day=1) - timedelta(days=1)

    def matches(self, txn):
        date_int = txn.getDateInt()
        trans_date = convert_date_int_to_date_object(date_int)

        return self.full_month_range_start_day <= trans_date <= self.full_month_range_end_day

    def matchesAll(self):
        return False


# class ExampleExtension:
#     myContext = None
#     myExtensionObject = None
#     name = "Your Extension Name Here"
#     # The initialize method is called when the extension is loaded and provides the
#     # extension's context.  The context implements the methods defined in the FeatureModuleContext:
#     # http://infinitekind.com/dev/apidoc/com/moneydance/apps/md/controller/FeatureModuleContext.html
#     def initialize(self, extension_context, extension_object):
#         self.myContext = extension_context
#         self.myExtensionObject = extension_object
#
#         # here we register ourselves with a menu item to invoke a feature
#         # (ignore the button and icon mentions in the docs)
#         self.myContext.registerFeature(extension_object, "doSomethingCool", None, "Do Something Cool!")
#
#     # invoke(eventstring) is called when we receive a callback for the feature that
#     # we registered in the initialize method
#     def invoke(self, eventString=""):
#         self.myContext.setStatus("Python extension received command: %s" % (eventString))
#
#     # invoke(eventstring) is called when we receive a callback for the feature that
#     # we registered in the initialize method
#     def __str__(self):
#         return "ExampleExtension"


class Category(object):
    def __init__(self, account):
        self.account = account
        self.account_id = account.getUUID()
        self.account_name = account.getFullAccountName()
        self.account_type = account.getAccountType()

    @property
    def account_path(self):
        return [
            self.__class__(a)
            for a in self.account.getPath()
        ]

    def __hash__(self):
        return hash(self.account_id)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_dict(self):
        return {
            'account_id': self.account_id,
            'account_name': self.account_name,
            'account_type': str(self.account_type)
        }

    def __repr__(self):
        return '<Category id: {self.account_id} name: {self.account_name} type: {self.account_type}>'.format(self=self)


class Transaction(object):
    def __init__(self, txn):
        self.txn = txn
        self.transaction_id = txn.getUUID()
        self.account = Category(txn.getAccount())

    @property
    def date(self):
        return convert_date_int_to_date_object(self.txn.getDateInt())

    @property
    def other_transaction_count(self):
        return self.parent_transaction.txn.getOtherTxnCount()

    @property
    def other_transactions(self):
        return [
            self.__class__(self.parent_transaction.txn.getOtherTxn(i))
            for i in range(self.parent_transaction.other_transaction_count)
        ]

    @property
    def other_transaction_account_types(self):
        return {
            t.account.account_type
            for t in self.other_transactions
        }

    @property
    def is_pre_payment_transfer(self):
        if self.other_transaction_count <= 2:
            return False

        return {
            Account.AccountType.CREDIT_CARD,
            Account.AccountType.EXPENSE,
            Account.AccountType.BANK
        }.issubset(self.other_transaction_account_types)

    @property
    def is_inter_account_transfer(self):
        root_account = self.account.account_path[0]
        other_transaction_root_accounts = {
            t.account.account_path[0]
            for t in self.other_transactions
        }
        return {root_account} == other_transaction_root_accounts

    def __repr__(self):
        return repr(self.txn)

    @property
    def parent_transaction(self):
        return self.__class__(self.txn.getParentTxn())

    def to_dict(self):
        return {
            'account_id': self.account.account_id,
            'date': self.date.isoformat(),
            'description': self.txn.getDescription(),
            'transaction_id': self.transaction_id,
            'value': unicode(convert_value_to_decimal(self.txn.getValue())),
        }

if __name__ == '__main__':
   run_script()

# setting the "moneydance_extension" variable tells Moneydance to register that object
# as an extension 
# moneydance_extension = ExampleExtension()
