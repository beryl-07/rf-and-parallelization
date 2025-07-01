CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -pthread

SRC = main.cpp DecisionTree.cpp RandomForest.cpp
OBJ = $(patsubst %.cpp,build/%.o,$(SRC))

all: digit_rf

# Génère les images PNG à partir des fichiers DOT
.PHONY: pngs
dot2png:
	@mkdir -p trees/png
	@for f in trees/dot/*.dot; do \
	  [ -e "$$f" ] || continue; \
	  b=$$(basename "$$f" .dot); \
	  dot -Tpng "$$f" -o "trees/png/$$b.png"; \
	done

# Compilation

digit_rf: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: %.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p build

# Exécution du programme : make run
.PHONY: run
run: digit_rf
	./digit_rf

# Génération d'un dataset CSV : make dataset n=200 noise=0.1 out=dataset.csv
.PHONY: dataset
dataset:
	$(CXX) $(CXXFLAGS) -o generate_dataset generate_dataset.cpp
	./generate_dataset $${n:-500} $${noise:-0.1} $${out:-dataset/dataset.csv}

clean:
	rm -rf build digit_rf trees/dot/*.dot trees/png/*.png
