"""RSS/NSE/BSE/GDELT parsing, ingestion, deduplication, entity linking (spec 10.1)."""

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from tradepartner.db.models import Instrument, NewsEntity, NewsRaw
from tradepartner.news import ingest as ing
from tradepartner.news.entity_link import AliasTable, normalize_name
from tradepartner.news.sources import base, rss
from tradepartner.news.sources.base import RawArticle
from tradepartner.news.sources.bse import BseAnnouncements
from tradepartner.news.sources.gdelt import GdeltSource
from tradepartner.news.sources.nse import NseAnnouncements

NOW = datetime(2026, 3, 10, 5, 0, tzinfo=UTC)
NAMES = {"INFY": "Infosys Limited", "TCS": "Tata Consultancy Services Limited",
         "RELIANCE": "Reliance Industries Limited", "ITC": "ITC Limited", "TATAMOTORS": "Tata Motors Limited",
         "TATASTEEL": "Tata Steel Limited", "M&M": "Mahindra & Mahindra Limited"}  # fmt: skip
ISINS = {"INE009A01021": "INFY"}
EXTRA = {"infosys": ["INFY"], "tata": ["TATAMOTORS", "TATASTEEL", "TCS"]}


@pytest.fixture
def aliases() -> AliasTable:
    return AliasTable(NAMES, ISINS, EXTRA)


def art(
    title: str, url: str = "", body: str = "", when: datetime | None = NOW, **kw: object
) -> RawArticle:
    return RawArticle("test", url or f"https://x/{abs(hash(title))}", title, body, when, **kw)  # type: ignore[arg-type]


# ---------- entity linking ----------


def test_name_normalization() -> None:
    assert normalize_name("Infosys Limited") == "infosys"
    assert normalize_name("Reliance Industries Ltd.") == "reliance industries"
    assert normalize_name("State Bank of India") == "state bank"


def test_alias_ticker_isin_and_title_before_body(aliases: AliasTable) -> None:
    m, amb = aliases.match("Infosys wins large deal", "TCS also bid")
    assert [(x.symbol, x.confidence) for x in m] == [("INFY", 0.9)] and amb == []  # body ignored
    m, _ = aliases.match("Quarterly update", "Reliance Industries posts record profit")
    assert [(x.symbol, x.confidence) for x in m] == [("RELIANCE", 0.6)]
    m, _ = aliases.match("INFY shares rise", "")
    assert m[0].symbol == "INFY" and m[0].method == "alias"
    m, _ = aliases.match("Stake sale", "ISIN INE009A01021 allotment")
    assert m[0].symbol == "INFY" and m[0].method == "isin" and m[0].confidence == 0.99
    assert aliases.match("Markets close flat", "nothing to see") == ([], [])


def test_word_boundaries_and_stoplist(aliases: AliasTable) -> None:
    # 'infosys' is not found inside 'information'
    assert aliases.match("The information technology sector", "")[0] == []
    # 'IT' is stop-listed, never a ticker
    assert aliases.match("IT stocks fall", "")[0] == []
    # exact tickers (even with punctuation in them) do match
    assert [x.symbol for x in aliases.match("M&M launches new SUV", "")[0]] == ["M&M"]
    assert [x.symbol for x in aliases.match("Mahindra & Mahindra posts profit", "")[0]] == ["M&M"]


def test_ambiguous_alias_is_flagged_not_guessed(aliases: AliasTable) -> None:
    m, amb = aliases.match("Tata group plans new investment", "")
    assert m == [] and amb[0].candidates == ("TATAMOTORS", "TATASTEEL", "TCS")
    # an unambiguous match on a candidate resolves the ambiguity
    m, amb = aliases.match("Tata Motors and Tata group plan EV spend", "")
    assert [x.symbol for x in m] == ["TATAMOTORS"] and amb == []


def test_alias_table_from_database_and_csv(db: Session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    db.add(
        Instrument(
            instrument_token=1,
            exchange="NSE",
            tradingsymbol="INFY",
            name="INFOSYS",
            isin="INE009A01021",
            segment="NSE",
        )
    )
    db.add(
        Instrument(
            instrument_token=2, exchange="NSE", tradingsymbol="NONAME", name=None, segment="NSE"
        )
    )
    db.commit()
    csv = tmp_path / "aliases.csv"
    csv.write_text("# c\nalias,symbol\nthe it major,INFY\n")
    t = AliasTable.from_session(db, csv)
    assert t.match("The IT major beats estimates", "")[0][0].symbol == "INFY"
    assert t.match("NONAME rallies", "")[0][0].symbol == "NONAME"
    assert AliasTable.from_session(db, tmp_path / "missing.csv").match("Infosys up", "")[0]
    assert AliasTable().match("anything", "") == ([], [])


# ---------- parsing ----------

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Infosys &amp; partners</title><link>https://a/1</link><description>&lt;p&gt;Body &lt;b&gt;text&lt;/b&gt;&lt;/p&gt;</description>
<pubDate>Tue, 10 Mar 2026 10:30:00 +0530</pubDate></item>
<item><title>No date</title><link>https://a/2</link></item>
<item><title></title><link>https://a/3</link></item>
</channel></rss>"""
ATOM = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom story</title>
<link href="https://b/1"/><summary>sum</summary><updated>2026-03-10T04:00:00Z</updated></entry></feed>"""


def test_rss_and_atom_parsing() -> None:
    items = rss.parse_feed(RSS, "rss:x")
    assert [i.title for i in items] == ["Infosys & partners", "No date"]
    assert items[0].body == "Body text" and items[0].published_at == datetime(
        2026, 3, 10, 5, 0, tzinfo=UTC
    )
    assert items[1].published_at is None
    atom = rss.parse_feed(ATOM, "rss:y")
    assert atom[0].url == "https://b/1" and atom[0].published_at == datetime(
        2026, 3, 10, 4, 0, tzinfo=UTC
    )
    assert rss._parse_date("garbage") is None and rss._parse_date(None) is None
    assert rss._parse_date("2026-03-10T04:00:00").tzinfo is not None  # naive ISO is taken as UTC
    assert base.clean_text(None) == "" and base.clean_text("<b>a</b>  &amp;  b") == "a & b"


def test_rss_source_isolates_bad_feeds_and_blocks_entity_bombs() -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><rss><channel><item><title>&b;</title><link>https://z</link></item></channel></rss>'

    def handler(req: httpx.Request) -> httpx.Response:
        return {
            "/ok": httpx.Response(200, text=RSS),
            "/bad": httpx.Response(500),
            "/bomb": httpx.Response(200, text=bomb),
            "/junk": httpx.Response(200, text="<<<"),
        }[req.url.path]

    src = rss.RssSource(
        [f"https://h{p}" for p in ("/ok", "/bad", "/bomb", "/junk")],
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    got = src.fetch(NOW - timedelta(days=1))
    assert [a.title for a in got] == [
        "Infosys & partners",
        "No date",
    ]  # the other three failed safely
    assert src.fetch(NOW + timedelta(days=1)) == [a for a in got if a.published_at is None]


def test_nse_announcements_and_result_calendar() -> None:
    ann = [{"symbol": "INFY", "desc": "Outcome of Board Meeting", "attchmntText": "Results approved", "an_dt": "10-Mar-2026 10:15:00",
            "attchmntFile": "https://nse/f.pdf", "sm_isin": "INE009A01021"},
           {"symbol": "", "desc": "x"}]  # fmt: skip
    cal = [{"symbol": "TCS", "purpose": "Financial Results/Dividend", "date": "12-Mar-2026"},
           {"symbol": "ITC", "purpose": "Fund raising", "date": "12-Mar-2026"}]  # fmt: skip
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/corporate-announcements":
            return httpx.Response(200, json=ann)
        if req.url.path == "/api/event-calendar":
            return httpx.Response(200, json=cal)
        return httpx.Response(200, text="ok")

    src = NseAnnouncements(httpx.Client(transport=httpx.MockTransport(handler)))
    got = src.fetch(datetime(2026, 3, 9, tzinfo=UTC))
    assert len(got) == 1 and got[0].symbol_hint == "INFY" and got[0].isin_hint == "INE009A01021"
    assert got[0].published_at == datetime(2026, 3, 10, 4, 45, tzinfo=UTC) and got[
        0
    ].title.startswith("INFY:")
    assert calls[0] == "/"  # warmed the cookie jar first
    assert src.fetch_result_calendar(date(2026, 3, 10), date(2026, 3, 20)) == [
        ("TCS", date(2026, 3, 12), "Financial Results/Dividend")
    ]
    fail = NseAnnouncements(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    )
    assert (
        fail.fetch(NOW) == []
        and fail.fetch_result_calendar(date(2026, 3, 1), date(2026, 3, 2)) == []
    )


def test_bse_and_gdelt_sources() -> None:
    bse_rows = {"Table": [{"HEADLINE": "Board meeting intimation", "SLONGNAME": "INFOSYS LTD", "NEWS_DT": "2026-03-10T10:00:00.123", "NSURL": "https://bse/1"},
                          {"HEADLINE": "", "NEWSSUB": ""}, {"HEADLINE": "No date", "NSURL": "https://bse/2", "NEWS_DT": "bad"}]}  # fmt: skip
    bse = BseAnnouncements(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bse_rows)))
    )
    got = bse.fetch(datetime(2026, 3, 9, tzinfo=UTC))
    assert [a.title for a in got] == ["INFOSYS LTD: Board meeting intimation", "No date"] and got[
        1
    ].published_at is None
    assert (
        BseAnnouncements(
            httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        ).fetch(NOW)
        == []
    )

    arts = {"articles": [{"url": "https://g/1", "title": "Infosys wins", "seendate": "20260310T043000Z", "language": "English"},
                         {"url": "https://g/2", "title": "bad date", "seendate": "xx"}]}  # fmt: skip
    seen: list[str] = []

    def gd(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.params["query"])
        return httpx.Response(200, json=arts)

    g = GdeltSource({'"Infosys"': "INFY"}, httpx.Client(transport=httpx.MockTransport(gd)))
    out = g.fetch_range(NOW - timedelta(days=7), NOW)
    assert len(out) == 1 and out[0].symbol_hint == "INFY" and "sourcecountry:IN" in seen[0]
    assert g.fetch(NOW)  # fetch() delegates to the range
    broken = GdeltSource(
        {'"x"': "X"}, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    )
    assert broken.fetch_range(NOW, NOW) == []


# ---------- ingestion ----------


def test_ingest_stores_dedupes_and_links(db: Session, aliases: AliasTable) -> None:
    batch = [
        art("Infosys wins mega deal", "https://x/1", "Details here"),
        art("Infosys wins mega deal", "https://x/1"),  # same URL
        art(
            "Totally different url", "https://x/2", "Details here"
        ),  # not the same title/body -> new
        art(
            "Infosys wins a mega deal", "https://x/3", "Other words entirely"
        ),  # near-duplicate title
        art("", "https://x/4"),
    ]
    res = ing.ingest(db, batch, NOW, aliases)
    assert (res.stored, res.dup_url, res.near_dup, res.skipped_empty) == (2, 1, 1, 1)
    assert db.scalar(select(func.count()).select_from(NewsRaw)) == 2
    links = {(e.symbol, e.match_method) for e in db.scalars(select(NewsEntity))}
    assert links == {("INFY", "alias")}
    again = ing.ingest(db, batch[:1], NOW, aliases)
    assert again.stored == 0 and again.dup_url == 1


def test_content_hash_dedupes_different_urls(db: Session) -> None:
    a = art("Same story", "https://x/a", "Body one")
    b = RawArticle("other", "https://y/b", "Same story", "Body one", NOW)
    res = ing.ingest(db, [a, b], NOW)
    assert res.stored == 1 and res.dup_hash == 1


def test_near_duplicate_keeps_the_earliest_and_respects_the_window(db: Session) -> None:
    first = art("RBI expected to cut rates next month say analysts", "https://x/1", when=NOW)
    ing.ingest(db, [first], NOW)
    later_dup = art(
        "RBI expected to cut rates next month, analysts say", "https://x/2",
        when=NOW + timedelta(hours=3),
    )  # fmt: skip
    far_later = art(
        "RBI expected to cut rates next month, analysts say", "https://x/3",
        when=NOW + timedelta(hours=30), body="a different body",
    )  # fmt: skip
    earlier = art(
        "RBI expected to cut rates next month, say the analysts", "https://x/4",
        when=NOW - timedelta(hours=2),
    )  # fmt: skip
    res = ing.ingest(db, [later_dup, far_later, earlier], NOW + timedelta(hours=31))
    assert res.near_dup == 1  # only the copy within 24 h AFTER the original is dropped
    assert res.stored == 2  # the 30 h repeat and the earlier-published copy are both kept


def test_published_at_estimation_and_future_dates(db: Session) -> None:
    res = ing.ingest(db, [art("No timestamp", "https://x/1", when=None), art("From the future", "https://x/2", when=NOW + timedelta(days=3)),
                          art("Fresh", "https://x/3", when=NOW - timedelta(hours=1))], NOW)  # fmt: skip
    rows = {r.title: r for r in db.scalars(select(NewsRaw))}
    assert res.stored == 3
    assert rows["No timestamp"].published_at_estimated and rows["No timestamp"].published_at == NOW
    assert (
        rows["From the future"].published_at_estimated
        and rows["From the future"].published_at == NOW
    )
    assert not rows["Fresh"].published_at_estimated and rows["Fresh"].fetched_at == NOW


def test_exchange_hints_ambiguity_and_append_only(db: Session, aliases: AliasTable) -> None:
    res = ing.ingest(db, [art("Board meeting", "https://n/1", symbol_hint="TCS", isin_hint="INE467B01029"),
                          art("Board meeting soon", "https://n/2", symbol_hint="ITC"),
                          art("Tata group mulls big capex", "https://n/3")], NOW, aliases)  # fmt: skip
    methods = {(e.symbol, e.match_method, e.confidence) for e in db.scalars(select(NewsEntity))}
    assert ("TCS", "isin", 1.0) in methods and ("ITC", "alias", 1.0) in methods
    assert len(res.ambiguous) == 1 and res.ambiguous[0][1][0].candidates[0] == "TATAMOTORS"
    with pytest.raises(DatabaseError, match="append-only"):
        db.execute(text("UPDATE news_raw SET title = 'tampered'"))
    db.rollback()


def test_normalize_and_hash_are_stable() -> None:
    assert ing.normalize("  Hello,   WORLD!! ") == "hello world"
    assert ing.content_hash("A b", "c") == ing.content_hash("a  B!", "C")


def test_boilerplate_titles_for_different_companies_are_not_near_duplicates(db: Session) -> None:
    batch = [
        art("ABB: Board Meeting Intimation for results", "https://n/1", symbol_hint="ABB"),
        art("TCS: Board Meeting Intimation for results", "https://n/2", symbol_hint="TCS"),
        art(
            "ABB: Board Meeting Intimation for results.",
            "https://n/3",
            "different body",
            symbol_hint="ABB",
        ),
    ]
    res = ing.ingest(db, batch, NOW)
    assert res.stored == 2 and res.near_dup == 1  # only the same company's repeat is dropped


def test_bse_non_json_object_responses_are_handled() -> None:
    for payload in ('"No Record Found!"', "{}", '{"Table": null}'):
        src = BseAnnouncements(
            httpx.Client(
                transport=httpx.MockTransport(lambda r, p=payload: httpx.Response(200, text=p))
            )
        )
        assert src.fetch(NOW) == []
