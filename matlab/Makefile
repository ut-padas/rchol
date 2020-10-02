
DIR = rchol_lap
MFILE = rchol_lap.mexa64

METIS_INC = 
METIS_LIB = 

# METIS is required for parallel
ifneq ($(and $(METIS_INC),$(METIS_LIB)),)
MFILE += metis_separator.mexa64
else
$(warning METIS_INC (metis header directory) and/or METIS_LIB (metis library directory) is not set)
$(warning rchol() cannot run in parallel)
endif


default: $(MFILE)

rchol_lap.mexa64: $(DIR)/rchol_lap.cpp $(DIR)/spcol.c
	mex $^

metis_separator.mexa64: $(DIR)/metis_separator.cpp
	mex $^ -I$(METIS_INC) -L$(METIS_LIB) -lmetis

clean:
	rm -f *.mexa64

