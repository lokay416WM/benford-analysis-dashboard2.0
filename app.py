import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import io

# Page configuration
st.set_page_config(
    page_title="Benford Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard title
st.title("Benford Law Analysis Dashboard")

st.write(
    "Upload financial data to analyze numerical patterns "
    "using Benford's Law."
)

st.divider()

# File upload section
st.header("1. Upload Your Data")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

# Read uploaded file
if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

            st.success("File uploaded successfully!")

            st.subheader("2. Preview Your Data")
            st.dataframe(df.head(10))

            st.write("Number of Rows:", len(df))
            st.write("Number of Columns:", len(df.columns))
            st.divider()

            st.subheader("3. Filter Your Data")

            df_filtered = df.copy()

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df_filtered = df.copy()

            st.success("File uploaded successfully!")

            # Filter 1: Numeric range
            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_columns:
                numeric_column = st.selectbox(
                    "Filter by numeric amount",
                    numeric_columns
                )

                minimum = float(df[numeric_column].min())
                maximum = float(df[numeric_column].max())

                if minimum < maximum:
                    amount_range = st.slider(
                        "Select amount range",
                        max_value=maximum,
                        value=(minimum, maximum)
                    )
                                            
                    df_filtered = df_filtered[
                        df_filtered[numeric_column].between(  
                            amount_range[0], amount_range[1]
                    )
                ]

        # filter 2: Category selection
        category_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if category_columns:
            category_column = st.selectbox(
                "Choose a category column",
                category_columns
            )

            categories = df[category_column].dropna().unique()

            selected_categories = st.multiselect(
                "Select categories",
                categories,
                default=list(categories)
            )

            df_filtered = df_filtered[
                df_filtered[category_column].isin(selected_categories)
            ]

        # Filter 3: Text search
        search_text = st.text_input(
            "Search descriptions or other text"
        )

        if search_text:
            text_columns = df_filtered.select_dtypes(
                include=["object", "category"]
            ).columns

            mask = df_filtered[text_columns].astype(str).apply(
                lambda col: col.str.contains(
                    search_text, case=False, na=False
                )
            ).any(axis=1)

            df_filtered = df_filtered[mask]

            st.write("Filtered Records:", len(df_filtered))
            st.dataframe(df_filtered.head(10))

        
        st.divider()
        st.header("4. Benford's Law Analysis")

        # Select the column to analyze
        benford_column = st.selectbox(
            "Select the column for Benford analysis",
            numeric_columns,
            key="benford_column"
        )

        # Convert data to numeric values
        amounts = pd.to_numeric(
            df_filtered[benford_column],
            errors="coerce"
        ).dropna()

        amounts = amounts[
            np.isfinite(amounts) & (amounts != 0)
        ].abs()

        if len(amounts) < 2:
            st.warning(
                "Not enough nonzero numeric values to perform analysis."
            )
        else:
            # Extract leading digits
            exponents = np.floor(np.log10(amounts))
            scaled = amounts / (10 ** exponents)

            first_digits = np.floor(scaled).astype(int)
            first_two_digits = np.floor(
                scaled * 10
            ).astype(int)

            # First digit Benford probabilities
            first_values = np.arange(1, 10)

            first_expected = np.log10(
                1 + 1 / first_values
            )

            first_actual = np.array([
                np.mean(first_digits == d)
                for d in first_values
            ])

            # First two digits Benford probabilities
            two_values = np.arange(10, 100)

            two_expected = np.log10(
                1 + 1 / two_values
            )

            two_actual = np.array([
                np.mean(first_two_digits == d)
                for d in two_values
            ])

            # Display first digit results
            st.subheader("First Digit Analysis")

            first_results = pd.DataFrame({
                "Digit": first_values,
                "Actual Percentage": first_actual * 100,
                "Benford Expected Percentage": first_expected * 100
            })

            st.dataframe(first_results, hide_index=True)

            # Display first two digit results
            st.subheader("First Two Digits Analysis")

            two_results = pd.DataFrame({
                "First Two Digits": two_values,
                "Actual Percentage": two_actual * 100,
                "Benford Expected Percentage": two_expected * 100
            })

            st.dataframe(two_results, hide_index=True)

            # KS statistical tests
            st.subheader("Kolmogorov-Smirnov (KS) Statistical Tests")

            first_cumulative = np.concatenate([
                [0], np.cumsum(first_expected)
            ])

            two_cumulative = np.concatenate([
                [0], np.cumsum(two_expected)
            ])

            first_test = kstest(
                first_digits,
                lambda x: first_cumulative[
                    np.searchsorted(
                        first_values, np.floor(x), side="right"
                    )
                ]
            )

            two_test = kstest(
                first_two_digits,
                lambda x: two_cumulative[
                    np.searchsorted(
                        two_values, np.floor(x), side="right"
                    )
                ]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("First Digit KS Test")
                st.metric(
                    "KS Statistic",
                    f"{first_test.statistic:.4f}"
                )
                st.metric(
                    "P-value",
                    f"{first_test.pvalue:.6f}"
                )

            with col2:
                st.write("First Two Digits KS Test")
                st.metric(
                    "KS Statistic",
                    f"{two_test.statistic:.4f}"
                )
                st.metric(
                    "P-value",
                    f"{two_test.pvalue:.6f}"
                )

            st.caption(
                "Note: The KS test uses a discrete Benford distribution. "
                "Its p-values are approximate because the digit data "
                "are discrete."
            )

            # First digit visualization
            st.subheader("First Digit Distribution")

            fig1, ax1 = plt.subplots(figsize=(10, 5))

            ax1.bar(
                first_values - 0.2,
                first_actual * 100,
                width=0.4,
                label="Actual"
            )

            ax1.bar(
                first_values + 0.2,
                first_expected * 100,
                width=0.4,
                label="Benford Expected"
            )

            ax1.set_xlabel("First Digit")
            ax1.set_ylabel("Percentage")
            ax1.set_title("Actual vs. Expected First Digit Distribution")
            ax1.set_xticks(first_values)
            ax1.legend()

            st.pyplot(fig1)
            plt.close(fig1)

            # First two digit visualization
            st.subheader("First Two Digits Distribution")

            fig2, ax2 = plt.subplots(figsize=(14, 6))

            ax2.plot(
                two_values,
                two_actual * 100,
                label="Actual",
                marker="."
            )

            ax2.plot(
                two_values,
                two_expected * 100,
                label="Benford Expected"
            )

            ax2.set_xlabel("First Two Digits")
            ax2.set_ylabel("Percentage")
            ax2.set_title("Actual vs. Expected First Two Digit Distribution")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig2)
            plt.close(fig2)


            # Downloadable reports
            st.divider()
            st.header("5. Download Analysis Results")

            # Summary statistics
            summary = amounts.describe().to_frame(
                name="Transaction Amount"
            )

            st.subheader("Summary Statistics")
            st.dataframe(summary)

            # Create Excel report
            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer, engine="openpyxl"
            ) as writer:

                summary.to_excel(
                    writer, sheet_name="Summary Statistics"
                )

                first_results.to_excel(
                    writer, sheet_name="First Digit",
                    index=False
                )

                two_results.to_excel(
                    writer, sheet_name="First Two Digits",
                    index=False
                )

            st.download_button(
                label="Download Complete Excel Report",
                data=excel_buffer.getvalue(),
                file_name="Benford_Analysis_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Download first digit chart
            chart_buffer = io.BytesIO()
            fig1.savefig(
                chart_buffer, format="png", bbox_inches="tight"
            )

            st.download_button(
                label="Download First Digit Chart",
                data=chart_buffer.getvalue(),
                file_name="First_Digit_Chart.png",
                mime="image/png"
            )

            # Download first two digit chart
            chart_buffer2 = io.BytesIO()
            fig2.savefig(
                chart_buffer2, format="png", bbox_inches="tight"
            )

            st.download_button(
                label="Download First Two Digits Chart",
                data=chart_buffer2.getvalue(),
                file_name="First_Two_Digit_Chart.png",
                mime="image/png"
            )

            if selected_categories:
                df_filtered = df_filtered[
                    df_filtered[category_column].isin(selected_categories)
                ]

    except Exception as e:
        st.error(f"Error reading file: {e}")

            
       