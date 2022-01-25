
Path registration for Linux, macOS and Windows Subsystem Linux (Optional)
  If you register the path of nsml, then you can call the client anywhere.
```
vim ~/.bashrc

.....

    export PATH=$PATH:[The path of nsml]

:wq

source ~/.bashrc
```

Login
  For Linux or macOS
    nsml login

  For Windows Command prompt
    nsml login -e CP949

  For Windows Subsystem Linux
    nsml.exe login -u 'YourID' -p 'YourPassword'
    (It is recommended to use quote '~~' for your password because of the whitespace or special characters.)

Commands
    nsml --help

