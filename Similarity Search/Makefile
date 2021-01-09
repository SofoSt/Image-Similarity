.PHONY: clean


CC = g++

CC_FLAGS = -g -std=c++11  -Wno-reorder -Ofast

BUILD_DIR = ./build
EXEC_DIR = ./executables

# Compile all
all: lsh cube cluster
	@echo all ready


#including these .mk files for lsh cube and clustering
include lsh.mk

include hypercube.mk

include clustering.mk

# Compilation
lsh: $(LSH_EXEC_DIR)/$(LSH_TARGET_EXEC)
	@echo lsh ready

cube: $(HC_EXEC_DIR)/$(HC_TARGET_EXEC)
	@echo cube ready

cluster: $(CL_EXEC_DIR)/$(CL_TARGET_EXEC)
	@echo cluster ready

# Cleaning
clean:
	$(RM) -r $(BUILD_DIR) $(EXEC_DIR)
clean_lsh:
	$(RM) -r $(BUILD_DIR)/lsh $(EXEC_DIR)/lsh
clean_cube:
	$(RM) -r $(BUILD_DIR)/hypercube $(EXEC_DIR)/hypercube
clean_cluster:
	$(RM) -r $(BUILD_DIR)/clustering $(EXEC_DIR)/clustering

# Default Arguments for path I am in
LSH_ARGS  = -d ../misc/original_space/train_set/t5k-images-idx3-ubyte -q ../misc/original_space/test_set/t1k-images-idx3-ubyte -i ../misc/reduced_space/train_reduced_5k -s ../misc/reduced_space/test_reduced_1k -k 4 -L 4 -o executables/lsh/lsh_out -N 1
LSH_ARGS_BIG  = -d ../misc/original_space/train_set/train-images-idx3-ubyte -q ../misc/original_space/test_set/t10k-images-idx3-ubyte -i ../misc/reduced_space/train_reduced_60k -s ../misc/reduced_space/test_reduced_10k -k 4 -L 4 -o executables/lsh/lsh_out -N 1


CUBE_ARGS = -d misc/datasets/train-images-idx3-ubyte  -q misc/querysets/t10k-images-idx3-ubyte -k 14 -M 3000 -probes 100 -o executables/hypercube/cube_out -N 1 -R 10000

CLUSTER_LLOYDS_ARGS = -d ../misc/original_space/train_set/t5k-images-idx3-ubyte -c config/cluster.conf -o executables/clustering/cluster_lloyds_out -i ../misc/reduced_space/train_reduced_5k -n ../misc/classes_for_clustering_small
CLUSTER_LLOYDS_ARGS_BIG = -d ../misc/original_space/train_set/train-images-idx3-ubyte -c config/cluster.conf -o executables/clustering/cluster_lloyds_out -i ../misc/reduced_space/train_reduced_60k -n ../misc/classes_for_clustering

# Running
run_lsh: lsh
	./$(EXEC_DIR)/lsh/lsh $(LSH_ARGS)
run_lsh_big: lsh
	./$(EXEC_DIR)/lsh/lsh $(LSH_ARGS_BIG)

run_cube: cube
	./$(EXEC_DIR)/hypercube/cube $(CUBE_ARGS)

run_cluster_lloyds: cluster
	./$(EXEC_DIR)/clustering/cluster $(CLUSTER_LLOYDS_ARGS)
run_cluster_lloyds_big: cluster
	./$(EXEC_DIR)/clustering/cluster $(CLUSTER_LLOYDS_ARGS_BIG)



# Valgrind
valgrind_lsh: lsh
	valgrind ./$(EXEC_DIR)/lsh/lsh $(LSH_ARGS)
valgrind_cube: cube
	valgrind ./$(EXEC_DIR)/hypercube/cube $(CUBE_ARGS)
valgrind_cluster_lloyds: cluster
	valgrind ./$(EXEC_DIR)/clustering/cluster $(CLUSTER_LLOYDS_ARGS)
	

-include $(DEPS)

MKDIR_P ?= mkdir -p