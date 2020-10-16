import configparser
import logging
import pandas
import sys

from jira import JIRA
from prettytable import PrettyTable
from settings import LOG_FILE_NAME
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger()


def configure_logging():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')
    stdout_logger_handler = logging.StreamHandler(sys.stdout)
    stdout_logger_handler.setLevel(logging.INFO)
    stdout_logger_handler.setFormatter(formatter)
    logger.addHandler(stdout_logger_handler)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')
    file_logger_handler = logging.FileHandler(LOG_FILE_NAME)
    file_logger_handler.setLevel(logging.NOTSET)
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)


def init_jira(config_file):
    logger.info('Initializing connection to Jira.')
    config = configparser.ConfigParser()
    config.read(config_file)

    server = config.get('JIRA', 'server')
    login = config.get('JIRA', 'login')
    password = config.get('JIRA', 'password')

    return JIRA(options={'server': server}, basic_auth=(login, password))


def get_train_data(jira):
    logger.info('Retrieving list with information about resolved bugs from Jira.')
    issues_list = _search_issues(
        jira,
        'type = Bug AND status IN (Resolved, Verified, Closed, Done, Complete) '
        'AND assignee IS NOT EMPTY AND component IS NOT Empty'
    )
    x_train = pandas.DataFrame([_get_x_structure(issue) for issue in issues_list])
    y_train = pandas.DataFrame([_get_y_structure(issue) for issue in issues_list])
    return x_train, y_train


def get_test_data(jira, bug_list):
    logger.info('Retrieving information about specified bugs from Jira.')
    issues_list = _search_issues(
        jira, 'type = Bug AND key IN ({}) AND assignee IS NOT EMPTY'.format(','.join(bug_list))
    )

    key_test = [_get_key_structure(issue) for issue in issues_list]
    x_test = pandas.DataFrame([_get_x_structure(issue) for issue in issues_list])
    return key_test, x_test


def preparing_data(x_train_raw, x_test_raw):
    logger.info('Preparing retrieved information from Jira for analyzing.')
    assignee_train, assignee_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'assignee')
    reporter_train, reporter_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'reporter')
    priority_train, priority_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'priority')
    components_train, components_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'components')
    project_train, project_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'project')
    testtype_train, testtype_test = _transform_categorical_signs(x_train_raw, x_test_raw, 'testtype')

    x_train = pandas.concat(
        [assignee_train, reporter_train, priority_train, components_train, project_train, testtype_train], axis=1
    )
    x_test = pandas.concat(
        [assignee_test, reporter_test, priority_test, components_test, project_test, testtype_test], axis=1
    )

    return x_train, x_test


def predict(x_train, y_train, x_test):
    logger.info('Predicting time resolution for specified bugs.')
    clf = RandomForestRegressor(n_estimators=50, random_state=1)
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def print_result(key_data, y_test):
    table = PrettyTable()
    table.field_names = ['Key', 'Predicted time resolution (hour)']
    for field in table.field_names:
        table.align[field] = 'l'

    for i in range(len(y_test)):
        table.add_row([key_data[i]['key'], round(y_test[i] / 3600.0, 1)])

    print(table)


def _get_x_structure(issue):
    return {
        'assignee': str(issue.fields.assignee),
        'reporter': str(issue.fields.reporter),
        'priority': str(issue.fields.priority),
        'components': str(issue.fields.components[0]),
        'project': str(issue.fields.project),
        'testtype': str(issue.fields.customfield_10127)
    }


def _get_y_structure(issue):
    return {
        'timespent': float(issue.fields.timespent) if issue.fields.timespent is not None else 0.0
    }


def _get_key_structure(issue):
    return {
        'key': str(issue.key)
    }


def _search_issues(jira, query):
    return jira.search_issues(
        query, maxResults=False,
        fields=['assignee', 'reporter', 'priority', 'components', 'project', 'customfield_10127', 'timespent']
    )


def _transform_categorical_signs(data_train, data_test, field_name):
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)

    data_train[field_name + '_le'] = le.fit_transform(data_train[field_name])
    data_test[field_name + '_le'] = le.transform(data_test[field_name])
    train_ohe = ohe.fit_transform(data_train[field_name + '_le'].values.reshape(-1, 1))
    new_train = pandas.DataFrame(train_ohe, columns=[field_name + '_' + str(i) for i in range(train_ohe.shape[1])])
    test_ohe = ohe.transform(data_test[field_name + '_le'].values.reshape(-1, 1))
    new_test = pandas.DataFrame(test_ohe, columns=[field_name + '_' + str(i) for i in range(test_ohe.shape[1])])

    return new_train, new_test
