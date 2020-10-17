import argparse
import logging
import sys

from settings import CONFIG_FILE, LOG_FILE_NAME, KEY_ISSUE_REGEX
from utils import configure_logging, configuring_parameters, get_train_data, get_test_data, preparing_data, predict, \
    print_result, validate, ValidateException


logger = logging.getLogger()


def main():
    configure_logging()
    logger.info('The tool was started: %s %s' % (sys.argv[0], ' '.join(['\'%s\'' % arg for arg in sys.argv[1:]])))
    try:
        bug_ids, config = parse_options()
        run(bug_ids, config)
    except ValidateException as e:
        logger.info(str(e))
        return 1
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
        '--bug-ids', type=str, required=True, metavar='BUG_IDS', help='List with bug IDs from Jira.'
    )
    parser.add_argument(
        '--config', required=False, metavar='CONFIG_FILE', default=CONFIG_FILE, help='Path to the config file.'
    )

    options = parser.parse_args()
    bug_ids = [x.strip() for x in options.bug_ids.split(',')]
    validate(
        bug_ids, KEY_ISSUE_REGEX, 'The following bug IDs are not valid: {}. Please, check its and re-run the tool.'
    )

    return bug_ids, options.config


def run(bug_ids, config):
    jira = configuring_parameters(config)
    x_train_raw, y_train = get_train_data(jira)
    key_test, x_test_raw, = get_test_data(jira, bug_ids)
    x_train, x_test = preparing_data(x_train_raw, x_test_raw)
    y_test = predict(x_train, y_train, x_test)
    print_result(key_test, y_test)


if __name__ == '__main__':
    sys.exit(main())
