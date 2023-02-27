################################################################################
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
# NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
# OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
# WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
# PURPOSE.
#
# NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences
# of use of such information or for any infringement of patents or other rights
# of third parties that may result from its use. No license is granted by
# implication or otherwise under any patent or patent rights of NVIDIA
# CORPORATION & AFFILIATES. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied. NVIDIA
# CORPORATION & AFFILIATES products are not authorized for use as critical
# components in life support devices or systems without express written approval
# of NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
#
################################################################################

CC := g++

MAIN := main.cpp
SRC_DIR := src
SRC_SM2_DIR := src/sm2
INC_SM2_DIR := include/sm2
OBJ_DIR := obj
OBJ_SM2_DIR := obj/sm2
BIN_DIR := bin
DEMO_DIR := ../samples/demo2/

EXE := $(BIN_DIR)/demo2
SRC := $(wildcard $(SRC_SM2_DIR)/*.cpp) $(wildcard $(SRC_SSM_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))
OBJ_SM2 := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))


CPPFLAGS := -Iinclude -Iinclude/ssm -I../../includes -Iinclude/sm2
CFLAGS   := -Wall
LDFLAGS  := -L/usr/local/driveworks/lib -Wl,-rpath,/usr/local/driveworks/lib
LDLIBS   :=

PYCMD := python3 ../../util/parser.py -i sm2.json -src ./src/sm2/ -inc ./include/sm2/

.PHONY: all clean parser

all: $(EXE) $(OBJ)

parser:
	$(PYCMD)

parser_int:
	 python3 ../../util/parser.py -i ./sm2.json -src ./src/sm2/ -inc ./include/sm2/

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ obj/main.o $(LDLIBS) -o $@ -lssm_release -lpthread

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@  $(CPPFLAGS)

$(OBJ_SM2_DIR)/%.o: $(SRC_SM2_DIR)/%.cpp | $(OBJ_SM2_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ $(CPPFLAGS)

$(OBJ_DIR):
	mkdir -p $(OBJ_SM2_DIR)

$(BIN_DIR):
	mkdir -p $@
	$(CC) $(CFLAGS) -c $(MAIN) -o obj/main.o  $(CPPFLAGS)


clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR) $(SRC_SM2_DIR) $(INC_SM2_DIR)

-include $(OBJ:.o=.d)

