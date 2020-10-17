# predict-bug-time-resolution
This tool needs to predict bug time resolution based on existing resolved bugs.

## How to prepare all the needed modules.
To install all required modules need to execute:
```
pip install -r requirements.txt
```

## How to run the script.
1. Create config.ini file by example sample.config.ini and specify access to Jira. 
2. Execute the following command:
```
python main.py --bug-ids <BUG_IDS> [--config CONFIG_FILE]
```

## How it works.
With machine learning, the system learns from already resolved bugs - all historical data kept in Jira. Based on how 
long the bug assignee resolved the previous bugs and on what project, this tool can predict the time resolution for the 
specified bug.

## Limitations.
1. For analyzing used custom field 'Test type'. Need to specify the field name in the settings.py file.
2. Only issues with a component will be analyzed. If no components specified for the issue, this issue will be ignored.
3. Only issues with an assignee will be analyzed. If no assignee specified for the issue, this issue will be ignored.
