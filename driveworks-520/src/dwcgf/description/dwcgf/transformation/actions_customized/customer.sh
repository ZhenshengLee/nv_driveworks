CUSTOMER_OUTPUT_PATH=~/customer
DWCGF_PATH=/usr/local/driveworks/src/dwcgf
# set env for dwcgf tool
export PYTHONPATH=$PYTHONPATH:$DWCGF_PATH
export PYTHONPATH=$PYTHONPATH:$DWCGF_PATH/dwcgf
# install missing modules, which are not included by default
pip3 install pyyaml
# create output directory
if [[ -d ${CUSTOMER_OUTPUT_PATH} ]]; then
    rm -rf ${CUSTOMER_OUTPUT_PATH}
fi
mkdir ${CUSTOMER_OUTPUT_PATH}
# customer
python3 ${DWCGF_PATH}/dwcgf/transformation/cli/transform.py ${DWCGF_PATH}/dwcgf/transformation/actions_customized/base_system/base_system.app.json -t ${DWCGF_PATH}/dwcgf/transformation/actions_customized/base_system/customized_actions.trans.json --actions=${DWCGF_PATH}/dwcgf/transformation/actions_customized/actions_customized.py -o ${CUSTOMER_OUTPUT_PATH}