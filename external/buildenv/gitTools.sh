###############################################################################
# Git Helpers for the build environment
###############################################################################

# Get the path
# Internal function that returns the current path if no argument is given.
function _get_path {
    path=$1
    if [ -z "${path}" ] ; then
        path=$(pwd)
    fi
    echo "$path"
}

# Check if path is a git repository
# returns a unix 0, 1 return code depending on the status
function git_is_repository {
    path=$(_get_path $1)
    git -C "${path}" rev-parse --is-inside-work-tree &> /dev/null
}

# Show if path is a repository
# echoes true or false
function git_repository {
    path=$(_get_path $1)
    if git_is_repository "${path}" ; then
        echo "true"
    else
        echo "false"
    fi
}

# Check the repository
function _check_path {
    path=$(_get_path $1)
    if ! git_is_repository "${path}" ; then
        echo "Not a git repository"
        exit 1
    fi
}

# Show the origin
# echoes git@github.com:pspoerri/cosmo-pompa.git
function git_show_origin {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    echo $(git -C "${path}" config --get remote.origin.url)
}

# Show the revision
# echoes cf28a6f
function git_show_revision {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    echo $(git -C "${path}" rev-parse --short HEAD)
}

# Show the check in date of the head
# echoes 2015-12-21 10:42:20 +0100
function git_show_checkindate {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    revision=$(git_show_revision "${path}")
    echo $(git -C "${path}" show -s --format=%ci ${revision})
}

# Show the current branch
# echoes buildenv
function git_show_branch {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    echo $(git -C "${path}" rev-parse --abbrev-ref HEAD)
}

# Show all the branch information and where the head is pointing
# echoes (HEAD -> buildenv, origin/buildenv)
function git_show_branch_all {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    echo $(git -C "${path}" log -n 1 --pretty=%d HEAD)
}

# Determines if the branch is dirty or not.
function git_repository_is_clean {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    # --quiet will invoke --exit-code which makes git exit with code 1 if there are
    # changes in the repository.
    git -C "${path}" diff --quiet &> /dev/null
}

# Determines the status of a repository
# echoes clean or dirty
function git_show_repository_status {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    if git_repository_is_clean "${path}" ; then
        echo "clean"
    else
        echo "dirty"
    fi
}

# Pretty print git info
# echoes "No git repository" if we are not dealing with a git repository
# echoes "Rev cf28a6f (dirty) on buildenv from git@github.com:pspoerri/cosmo-pompa.git"
# otherwise
function git_info {
    path=$(_get_path $1)
    if ! _check_path $1 ; then exit 1; fi
    revision=$(git_show_revision "${path}")
    branch=$(git_show_branch "${path}")
    origin=$(git_show_origin "${path}")
    dirty=""
    if ! git_repository_is_clean "${path}" ; then
        dirty=" ($(git_show_repository_status ${path}))"
    fi
    echo "Rev ${revision}${dirty} on ${branch} from ${origin}"
}

# Function to test the implementation
function test_functions {
    path=$(_get_path $1)
    echo "Checking the function with \"${path}\" as argument."
    echo "---------------------------------------------------------------------"
    echo "Origin       :" $(git_show_origin "${path}")
    echo "Revision     :" $(git_show_revision "${path}")
    echo "Check in date:" $(git_show_checkindate "${path}")
    echo "Branch       :" $(git_show_branch "${path}")
    echo "Branch all   :" $(git_show_branch_all "${path}")
    echo "Status       :" $(git_show_repository_status "${path}")
    echo "In repository:" $(git_repository "${path}")
    echo "Info         :" $(git_info "${path}")
    echo ""
    echo "Testing a unversioned folder"
    echo "---------------------------------------------------------------------"
    echo "Origin       :" $(git_show_origin /)
    echo "Revision     :" $(git_show_revision /)
    echo "Check in date:" $(git_show_checkindate /)
    echo "Branch       :" $(git_show_branch /)
    echo "Branch all   :" $(git_show_branch_all /)
    echo "Status       :" $(git_show_repository_status /)
    echo "In repository:" $(git_repository /)
    echo "Info         :" $(git_info /)
    echo ""
    echo "Checking without arguments"
    echo "---------------------------------------------------------------------"
    echo "Origin       :" $(git_show_origin)
    echo "Revision     :" $(git_show_revision)
    echo "Check in date:" $(git_show_checkindate)
    echo "Branch       :" $(git_show_branch)
    echo "Branch all   :" $(git_show_branch_all)
    echo "Status       :" $(git_show_repository_status)
    echo "In repository:" $(git_repository)
    echo "Info         :" $(git_info)
}

export -f git_show_origin
export -f git_show_revision
export -f git_show_checkindate
export -f git_show_branch
export -f git_show_branch_all
export -f git_repository_is_clean
export -f git_show_repository_status
export -f git_is_repository
export -f git_repository
export -f git_info
