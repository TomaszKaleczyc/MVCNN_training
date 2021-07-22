ENV_FOLDER=./environment
VENV_NAME=mvcnn_train
VENV_PATH=$(ENV_FOLDER)/$(VENV_NAME)
VENV_ACTIVATE_PATH=$(VENV_PATH)/bin/activate
REQUIREMENTS_PATH=$(ENV_FOLDER)/requirements.txt 

create-env:
	python3 -m virtualenv --system-site-packages -p python3.6 $(VENV_PATH)
	. $(VENV_ACTIVATE_PATH) && \
	python3 -m pip install pip --upgrade && \
	python3 -m pip install -r $(REQUIREMENTS_PATH)# && \
	# ipython kernel install --user --name=$(VENV_NAME)

activate-env-command:
	@echo source $(VENV_ACTIVATE_PATH)


run-tensorboard:
	tensorboard --logdir ./output/