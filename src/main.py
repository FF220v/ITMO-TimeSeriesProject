from src.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run_server(host="0.0.0.0", port=8050, debug=True)