FROM python:3.12
RUN mkdir -p /usr/src/install/logs/
RUN touch /usr/src/install/logs/install.txt
RUN echo 'Created log file' > /usr/src/install/logs/install.txt
RUN apt-get update
RUN echo 'Updated apt-get' >> /usr/src/install/logs/install.txt
RUN apt install -y neovim
RUN echo 'Installed neovim' >> /usr/src/install/logs/install.txt
RUN apt install -y curl
RUN echo "Installed curl" >> /usr/src/install/logs/install.txt
RUN apt-get install -y clang
RUN echo 'Installed clang' >> /usr/src/install/logs/install.txt
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
RUN echo 'Downloaded rustup' >> /usr/src/install/logs/install.txt
RUN sh rustup.sh -y >> /usr/src/install/logs/install.txt
RUN echo 'Installed rustup' >> /usr/src/install/logs/install.txt
RUN cat /usr/src/install/logs/install.txt
RUN echo "Home directory:" >> /usr/src/install/logs/install.txt
RUN echo ${HOME} >> /usr/src/install/logs/install.txt
ENV PATH="/root/.cargo/bin:${PATH}"
RUN echo 'Updated PATH' >> /usr/src/install/logs/install.txt
RUN echo "PATH:" >> /usr/src/install/logs/install.txt
RUN echo ${PATH} >> /usr/src/install/logs/install.txt
RUN cat /usr/src/install/logs/install.txt
RUN cargo --help >> /usr/src/install/logs/install.txt
RUN echo 'Checked cargo' >> /usr/src/install/logs/install.txt
RUN rustup toolchain install nightly
RUN echo 'Installed rust nightly' >> /usr/src/install/logs/install.txt
RUN rustup default nightly
RUN echo 'Set rust default to nightly' >> /usr/src/install/logs/install.txt
RUN cargo +nightly install hvm
RUN echo 'Installed hvm' >> /usr/src/install/logs/install.txt
RUN cargo +nightly install bend-lang
RUN echo 'Installed bend-lang' >> /usr/src/install/logs/install.txt
RUN apt install -y git
RUN echo 'Installed git' >> /usr/src/install/logs/install.txt
RUN pip install pipx
RUN echo 'Installed pipx' >> /usr/src/install/logs/install.txt
RUN pipx install hatch
ENV PATH="/root/.local/bin:${PATH}"
RUN echo 'Installed hatch' >> /usr/src/install/logs/install.txt
RUN git clone -b develop https://github.com/octakitten/opencb.git
RUN echo 'Cloned opencb' >> /usr/src/install/logs/install.txt
RUN echo ${PATH}
WORKDIR "/opencb"
RUN echo ${PATH}
RUN hatch build opencb
RUN echo 'Installed opencb dependencies' >> /usr/src/install/logs/install.txt


RUN echo 'Checking rust version' >> /usr/src/install/logs/install.txt
RUN rustc --version >> /usr/src/install/logs/install.txt
RUN echo 'Checking python version' >> /usr/src/install/logs/install.txt
RUN python3 --version >> /usr/src/install/logs/install.txt