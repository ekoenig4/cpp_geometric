SRC=src

.PHONY: all clean cleandict exe test proj


all: 
	$(MAKE) -C $(SRC)
cleandict: 
	$(MAKE) -C $(SRC) cleandict
clean: 
	$(MAKE) -C $(SRC) clean
exe: 
	$(MAKE) -C $(SRC) exe
test: 
	$(MAKE) -C $(SRC) test
	
proj: 
	mkdir -p bin
	mkdir -p obj
	mkdir -p lib