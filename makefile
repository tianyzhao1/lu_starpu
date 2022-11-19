CFLAGS += $$(pkg-config --cflags starpu-1.2)
CFLAGS += $$(pkg-config --cflags openblas)
LDLIBS += $$(pkg-config --libs starpu-1.2)
LDLIBS += $$(pkg-config --libs openblas)
main: main.o 
main.o: main.c



