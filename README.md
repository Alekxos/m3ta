# m3ta

Version of test-time training that uses MAML to learn from tasks at test-time. Uses the CIFAR-10 dataset with corruptions; dataset setup is described at the test-time CIFAR-10 repo (https://github.com/yueatsprograms/ttt_cifar_release).

To run the script with meta test-time training, use the command:
/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/script_mettta.sh
To run the original script with test-time training, use the command:/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/script_mettta.sh

The models exist in the models/ dir, and have been updated to facilitate backpropagation through an inner cycle of gradient computation with MAML. The implementation of MAML itself exists at test_calls/test_mettta.py. To switch between M3TA and TTT, it is currently necessary to respectively add or remove the initial dataset casting (`x = x.double()`).

Afterwards, use
python /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/show_table.py
to generate a table with the results comparing error per corruption task with M3TA versus with TTT.
