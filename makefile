CC = mpic++

SRC  = chrono.c
MAIN = mmul

CFLAGS = -I./
LDLIBS = -lmpi

all: $(SRC) $(MAIN)

clean:
	@ rm -f $(MAIN)

.PHONY: all clean
