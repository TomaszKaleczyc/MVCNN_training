VENV_NAME=mvcnn_train
VENV_PATH=./$(VENV_NAME)
VENV_ACTIVATE_PATH=$(VENV_PATH)/bin/activate

create-env:
	python3 -m virtualenv --system-site-packages -p python3.6 $(VENV_PATH)
	. $(VENV_ACTIVATE_PATH) && \
	python3 -m pip install pip --upgrade && \
	python3 -m pip install -r requirements.txt # && \
	# ipython kernel install --user --name=$(VENV_NAME)

activate-env:
	@echo source $(VENV_ACTIVATE_PATH)