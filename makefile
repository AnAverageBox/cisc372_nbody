FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.cu compute.cu
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f *.o nbody


#updated to allow for the new .cu files

#below is to keep the file extensions as .c
#FLAGS= -DDEBUG -x cu
#LIBS= -lm
#ALWAYS_REBUILD=makefile

# Note the change from .cu to .c here
#nbody: nbody.c compute.c
#	nvcc $(FLAGS) $^ -o $@ $(LIBS)

#lean:
#	rm -f *.o nbody