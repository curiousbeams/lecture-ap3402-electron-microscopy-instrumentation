# slim uv base image
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv /uv /uvx /bin/

# uv optimization env variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# install deps (system is safe, already isolated)
COPY ./requirements.txt .
RUN uv pip install -r requirements.txt

# create user with a home directory for binder
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# run jupyterlab
CMD ["jupyter", "lab"]
