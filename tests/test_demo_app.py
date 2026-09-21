from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_workflow():
    app=AppTest.from_file(APP,default_timeout=15).run()
    assert not app.exception and not app.error
    app.button[0].click().run()
    assert "Enter some Hindi text" in app.error[0].value
    app.selectbox[0].select("Person and place").run()
    assert "मीरा शर्मा" in app.text_area[0].value
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe)==1 and app.metric[0].value=="2"
    assert any("<mark " in m.value for m in app.markdown)
    assert len(app.get("download_button"))==2
    app.button[1].click().run()
    assert app.text_area[0].value=="" and len(app.dataframe)==0


def test_app_too_long_and_no_entities():
    app=AppTest.from_file(APP,default_timeout=15).run()
    app.text_area[0].input("अ"*5001).run()
    app.button[0].click().run()
    assert "5,000-character limit" in app.error[0].value
    app.selectbox[0].select("Everyday sentence").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("No entities detected" in x.value for x in app.info)
