import argparse
import logging
import sys

from settings import CONFIG_FILE, LOG_FILE_NAME
from utils import configure_logging, init_jira, get_train_data, get_test_data, preparing_data, predict, print_result


logger = logging.getLogger()


def main():
    configure_logging()
    logger.info('The tool was started: %s %s' % (sys.argv[0], ' '.join(['\'%s\'' % arg for arg in sys.argv[1:]])))
    try:
        bug_list, config = parse_options()
        run(bug_list, config)
    except SystemExit as e:
        logger.info('The tool finished work with the exit code {exit_code}.'.format(exit_code=e.code))
        return e.code
    except Exception as e:
        logger.debug('An unexpected error occurred: %s' % str(e), **{'exc_info': 1})
        logger.error('An unexpected error occurred. See the details in the log file "%s".' % LOG_FILE_NAME)
        return 1


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bug-list', type=str, required=True, metavar='BUG_LIST', help='List with bug IDs from Jira.'
    )
    parser.add_argument(
        '--config', required=False, metavar='CONFIG_FILE', default=CONFIG_FILE, help='Path to the config file.'
    )

    options = parser.parse_args()
    bug_list = [x.strip() for x in options.bug_list.split(',')]
    return bug_list, options.config


def run(bug_list, config):
    jira = init_jira(config)
    x_train_raw, y_train = get_train_data(jira)
    key_test, x_test_raw, = get_test_data(jira, bug_list)
    x_train, x_test = preparing_data(x_train_raw, x_test_raw)
    y_test = predict(x_train, y_train, x_test)
    print_result(key_test, y_test)


if __name__ == '__main__':
    sys.exit(main())
