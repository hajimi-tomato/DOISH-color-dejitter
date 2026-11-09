"""Browser launcher; also the entry point for the Windows executable."""
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    from app import main
    main(open_browser=True)
