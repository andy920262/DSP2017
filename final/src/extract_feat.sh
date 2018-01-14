#!/bin/bash

config=htk_config.cfg
training_list=scripts/training_list.scp
testing_list=scripts/testing_list.scp

HList -r -T 1 -C $config -S $training_list > data/train.dat
HList -r -T 1 -C $config -S $testing_list > data/test.dat
