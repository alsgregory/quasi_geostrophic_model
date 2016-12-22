all:

lint:
	@echo "    Linting source codebase"
	@flake8 quasi_geostrophic_model
	@echo "    Linting quasi-geostrophic test suite"
	@flake8 tests
	@echo "    Linting quasi-geostrophic demo suite"
	@flake8 demos

test:
	@echo "    Running quasi-geostrophic test suite"
	@py.test tests $(PYTEST_ARGS)
    
