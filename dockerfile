FROM python:3.12
RUN mkdir -p /usr/src/install/logs/
RUN touch /usr/src/install/logs/install.txt
RUN echo 'Created log file' > /usr/src/install/logs/install.txt
RUN apt-get update
RUN echo 'Updated apt-get' >> /usr/src/install/logs/install.txtx
RUN apt install -y curl
RUN echo "Installed curl" >> /usr/src/install/logs/install.txt
RUN apt-get install -y clang
RUN echo 'Installed clang' >> /usr/src/install/logs/install.txt
RUN cat /usr/src/install/logs/install.txt
RUN echo "Home directory:" >> /usr/src/install/logs/install.txt
RUN echo ${HOME} >> /usr/src/install/logs/install.txt
ENV PATH="/root/.cargo/bin:${PATH}"
RUN echo 'Updated PATH' >> /usr/src/install/logs/install.txt
RUN echo "PATH:" >> /usr/src/install/logs/install.txt
RUN echo ${PATH} >> /usr/src/install/logs/install.txt
RUN cat /usr/src/install/logs/install.txt
RUN apt install -y git
RUN echo 'Installed git' >> /usr/src/install/logs/install.txt
RUN pip install hatch
ENV PATH="/root/.local/bin:${PATH}"
RUN echo 'Installed hatch' >> /usr/src/install/logs/install.txt
RUN git clone -b develop https://github.com/octakitten/opencb.git
RUN echo 'Cloned opencb' >> /usr/src/install/logs/install.txt
RUN echo ${PATH}
WORKDIR "/opencb"
RUN echo ${PATH}
RUN hatch env prune
RUN hatch env create
RUN echo 'Created opencb environment' >> /usr/src/install/logs/install.txt
RUN echo 'Running Pytest' >> /usr/src/install/logs/install.txt
RUN hatch shell \
    && pytest >> /usr/src/install/logs/install.txt
RUN hatch build -t wheel dist/
RUN echo 'Installed opencb dependencies' >> /usr/src/install/logs/install.txt
RUN echo 'Checking python version' >> /usr/src/install/logs/install.txt
RUN python3 --version >> /usr/src/install/logs/install.txt