get(ENV, "TRAVIS_OS_NAME", "")       == "linux" || exit()

using Coverage

cd(joinpath(dirname(@__FILE__), "..")) do
    Coveralls.submit(process_folder())
end
