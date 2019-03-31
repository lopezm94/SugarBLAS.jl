get(ENV, "TRAVIS_OS_NAME", "")       == "linux" || exit()

using Coverage

cd(joinpath(dirname(@__FILE__), "..")) do
    Codecov.submit(Codecov.process_folder())
end
