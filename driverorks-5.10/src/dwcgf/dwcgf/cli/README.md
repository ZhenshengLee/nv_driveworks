# CGF Python CLI tools

## gdlAmend2appConfig

This tool is used to convert the exiting amend file to new CGF app-config file.

Note that it is only meant for converting the syntax, it's still the user's responsibility to make sure the correctness of the converted file, manual changes are required when necessary. Please pay attention to the console ouput for any warning/error
messages that indicate a manual inspection and comparasion between amend file and generated app-config file to guarantee the correct
behavior.

Example Usage:

1) Converting a single amend file providing an absolute path
```
dazel run //src/dwcgf/dwcgf/cli:gdlAmend2appConfig -- --amend_file=$HOME/ndas/apps/roadrunner-2.0/config/amends/noRadar.json
```
2) Converting all amend files inside a folder providing an absolute path
```
dazel run //src/dwcgf/dwcgf/cli:gdlAmend2appConfig -- --amend_folder=$HOME/ndas/partners/daimler/config/amends
```
The output app-config file will be created in the same directory where the input amend file is located.